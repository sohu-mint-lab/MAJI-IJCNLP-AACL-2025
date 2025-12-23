import json
import os
import math
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled, set_default_openai_api
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import asyncio
import warnings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict

# --- Environment and Client Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# --- Dataclasses for Data Loading ---

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation transcript."""
    speaker: str
    text: str

@dataclass
class OutlineQuestion:
    """Represents a single question within an outline section."""
    id: str
    question: str
    importance: float
    keywords: List[str] = field(default_factory=list)

@dataclass
class OutlineSection:
    """Represents a section of the interview outline."""
    id: str
    section: str
    questions: List[OutlineQuestion]

    def __post_init__(self):
        self.questions = [OutlineQuestion(**q) for q in self.questions]

# --- Dataclasses for Metrics ---

@dataclass
class QuestionMetrics:
    """Stores all computed metrics for a single suggested question."""
    question: str
    source: str
    was_selected: bool
    context_relevance: float
    outline_relevance: float
    originality: float
    qualitative_elaboration: float = 0.0

@dataclass
class VersionRoundResult:
    """Stores metrics for one version within a single round."""
    selected_suggestion_metrics: Optional[QuestionMetrics]
    all_suggestion_metrics: List[QuestionMetrics]

@dataclass
class CrossVersionEvaluationRound:
    """Stores metrics for a full round, comparing multiple versions."""
    round_num: int
    actual_question: str
    previous_answer: str
    results_by_version: Dict[str, VersionRoundResult] = field(default_factory=dict)

# --- Pydantic Models for AI Judge Outputs ---

class ElaborationJudgement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    elaboration: float = Field(..., description="Score for encouraging a detailed and elaborate response.")

# --- AI Judge Agent Definitions ---

ElaborationJudgeAgent = Agent(
    name="ElaborationJudgeAgent",
    instructions=(
        "You are an expert conversational analyst. Your task is to evaluate a single proposed interview question based on the conversation's context. "
        "Provide a score from 0.0 to 1.0 for the following quality:\n"
        "1.  **Elaboration**: Does the question encourage the interviewee to provide a detailed, in-depth, and comprehensive answer, rather than a short or simple one?\n\n"
        "Your output **MUST** be a single JSON object with the key: `elaboration`."
    ),
    model="gpt-4o-mini",
    output_type=ElaborationJudgement,
)

# --- Main Evaluator Class ---

class MajiEvaluator:
    """
    Evaluates and compares different versions of the DCAgent system
    based on pre-generated survey data.
    """

    def __init__(
        self,
        transcript_path: str,
        survey_data_paths: Dict[str, str], # e.g., {"v2": "path/to/v2.json", "v3": "path/to/v3.json"}
        outline_path: str,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.versions = list(survey_data_paths.keys())

        # --- Load Data ---
        print("--- Loading data and models... ---")
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            self.transcript = [ConversationTurn(**turn) for turn in transcript_data["transcript"]]

            self.survey_data: Dict[str, List[Dict]] = {}
            for version, path in survey_data_paths.items():
                with open(path, "r", encoding="utf-8") as f:
                    self.survey_data[version] = json.load(f)

            with open(outline_path, "r", encoding="utf-8") as f:
                outline_json = json.load(f)
            self.outline = [OutlineSection(**s) for s in outline_json]
            self.all_outline_questions = [q.question for s in self.outline for q in s.questions]

        except FileNotFoundError as e:
            print(f"Error: Could not find a required file: {e}")
            raise
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Error: Could not parse a required file. Check format. Details: {e}")
            raise

        # --- Load Models ---
        print("  [+] Loading embedding model...")
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"  [+] Models loaded on device: {self.device}")

        # --- Initialize Containers ---
        self.evaluation_results: List[CrossVersionEvaluationRound] = []

    # --- Core Calculation Methods ---

    def _cos_batch(self, main_str: str, str_list: List[str]) -> np.ndarray:
        if not main_str or not str_list: return np.array([0.0])
        main_str, str_list = main_str.strip(), [s.strip() for s in str_list if s and s.strip()]
        if not main_str or not str_list: return np.array([0.0])
        embeddings = self.similarity_model.encode([main_str] + str_list, normalize_embeddings=True)
        return np.dot(embeddings[1:], embeddings[0])

    # --- Main Evaluation Loop ---

    async def evaluate(self):
        print("\n=== Starting Cross-Version MAJI Evaluation ===")
        all_time_suggestions = [] # For originality calculation

        num_rounds = len(self.transcript) // 2
        for round_num in range(num_rounds):
            if self.verbose: print(f"\n[Round {round_num}] Processing...")
            
            actual_q_idx = round_num * 2
            if actual_q_idx >= len(self.transcript): continue
            
            actual_q_text = self.transcript[actual_q_idx].text.strip()
            prev_answer_text = self.transcript[actual_q_idx + 1].text.strip() if actual_q_idx + 1 < len(self.transcript) else ""
            
            round_evaluation = CrossVersionEvaluationRound(
                round_num=round_num,
                actual_question=actual_q_text,
                previous_answer=prev_answer_text
            )

            for version in self.versions:
                survey_item = next((item for item in self.survey_data[version] if item['round_num'] == round_num), None)
                if not survey_item: continue
                
                # *** FOCUS ON DCAGENT-ONLY SUGGESTIONS ***
                dc_suggestions = [s for s in survey_item['suggested_questions'] if s['source'].startswith('dcagent')]
                if not dc_suggestions: continue

                dc_suggestion_texts = [s['question'] for s in dc_suggestions]

                # --- Perform Per-Suggestion AI Analysis in Parallel ---
                analysis_tasks = []
                instruction = f"An interviewer is speaking with someone. The interviewee just said: '{prev_answer_text}'. The interviewer is considering asking the following question."
                for q_text in dc_suggestion_texts:
                    analysis_tasks.append(self._run_suggestion_analysis(instruction, q_text))
                
                analysis_results = await asyncio.gather(*analysis_tasks)
                analysis_map = {text: res for text, res in zip(dc_suggestion_texts, analysis_results)}

                # --- Calculate Metrics for each DCAgent suggestion ---
                qmetrics = []
                for sugg_dict in dc_suggestions:
                    q_text = sugg_dict['question'].strip()
                    if not q_text: continue
                    
                    ctx_rel = self._cos_batch(q_text, [prev_answer_text])[0] if prev_answer_text else 0.0
                    outline_rel = float(np.max(self._cos_batch(q_text, self.all_outline_questions)))
                    originality = 1.0 - (float(np.max(self._cos_batch(q_text, all_time_suggestions))) if all_time_suggestions else 0.0)
                    qual_res = analysis_map.get(q_text, {})

                    qmetrics.append(QuestionMetrics(
                        question=q_text,
                        source=sugg_dict['source'],
                        was_selected=sugg_dict['is_selected'],
                        context_relevance=float(ctx_rel),
                        outline_relevance=outline_rel,
                        originality=originality,
                        qualitative_elaboration=qual_res.get('elaboration', 0.0),
                    ))

                if not qmetrics: continue

                selected_metric = next((m for m in qmetrics if m.was_selected), None)

                round_evaluation.results_by_version[version] = VersionRoundResult(
                    selected_suggestion_metrics=selected_metric,
                    all_suggestion_metrics=qmetrics,
                )
            
            if round_evaluation.results_by_version:
                self.evaluation_results.append(round_evaluation)
                all_time_suggestions.extend([
                    q.question for res in round_evaluation.results_by_version.values() for q in res.all_suggestion_metrics
                ])

        print("\n=== Evaluation Complete ===")

    async def _run_suggestion_analysis(self, instruction: str, response: str) -> Dict:
        prompt = f"{instruction}\\n\\nPROPOSED QUESTION: {response}"
        try:
            runner_result = await Runner.run(ElaborationJudgeAgent, prompt)
            judgement = runner_result.final_output_as(ElaborationJudgement)
            if judgement:
                return judgement.model_dump()
        except Exception as e:
            if self.verbose: print(f"  -> Error during ElaborationJudgeAgent analysis: {e}")
        return {}

    # --- Reporting and Saving ---
    
    def _avg(self, values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
        
    def get_evaluation_summary(self) -> Dict:
        summary = {version: {} for version in self.versions}

        for version in self.versions:
            all_suggestions = [q for r in self.evaluation_results for q in r.results_by_version.get(version, VersionRoundResult(None, [])).all_suggestion_metrics]
            selected_suggestions = [r.results_by_version.get(version, VersionRoundResult(None, [])).selected_suggestion_metrics for r in self.evaluation_results if r.results_by_version.get(version, VersionRoundResult(None, [])).selected_suggestion_metrics]
            
            def get_quality_stats(questions: List[QuestionMetrics]):
                if not questions: return {}
                return {
                    "avg_elaboration": self._avg([q.qualitative_elaboration for q in questions]),
                    "avg_originality": self._avg([m.originality for m in questions]),
                    "avg_context_relevance": self._avg([m.context_relevance for m in questions]),
                    "avg_outline_relevance": self._avg([m.outline_relevance for m in questions]),
                }
            
            summary[version]["all_suggestions_analysis"] = get_quality_stats(all_suggestions)
            summary[version]["selected_questions_analysis"] = get_quality_stats(selected_suggestions)

        return summary

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = {
            "evaluation_summary": self.get_evaluation_summary(),
            "detailed_metrics_by_round": [asdict(r) for r in self.evaluation_results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nEvaluation results saved to: {output_path}")

async def main():
    """Main function to run the cross-version evaluation process."""
    transcript_file = "mermaid_real.json"
    
    transcript_path = f"data/transcripts/{transcript_file}"
    outline_path = "data/outlines/example_outline.json"
    
    survey_data_paths = {
        "v2": f"data/evaluations/survey_data_v2_{transcript_file}",
        "v3": f"data/evaluations/survey_data_v3_{transcript_file}"
    }
    output_path = f"data/evaluations/evaluation_results_maji_v2_v3_{transcript_file}"

    for v, path in survey_data_paths.items():
        if not os.path.exists(path):
            print(f"Error: Survey data file for version '{v}' not found at '{path}'")
            print("Please run 'maji_survey_data_generator.py' first to generate it.")
            return

    evaluator = MajiEvaluator(
        transcript_path=transcript_path,
        survey_data_paths=survey_data_paths,
        outline_path=outline_path,
        verbose=True,
    )
    await evaluator.evaluate()
    evaluator.save_results(output_path)
    
    summary = evaluator.get_evaluation_summary()
    print("\n--- Cross-Version MAJI Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    print("-----------------------------------------")

if __name__ == "__main__":
    asyncio.run(main()) 
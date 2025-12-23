import json
import os
import math
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled, set_default_openai_api
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import asyncio
import warnings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ConfigDict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

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
        # Ensure nested questions are also dataclass objects
        self.questions = [OutlineQuestion(**q) for q in self.questions]

# --- Dataclasses for Metrics ---

@dataclass
class QuestionMetrics:
    """Stores all computed metrics for a single suggested question."""
    question: str
    source: str
    was_selected: bool

    # Core Quality Metrics
    context_relevance: float  # Similarity to previous answer
    outline_relevance: float  # Similarity to the interview outline
    originality: float          # Replaces 'novelty'
    qualitative_elaboration: float = 0.0 # NEW: Does it encourage a detailed response?

@dataclass
class EvaluationRound:
    """Stores metrics for a full round of suggestions."""
    round_num: int
    actual_question: str
    previous_answer: str
    selected_suggestion_metrics: QuestionMetrics
    all_suggestion_metrics: List[QuestionMetrics]

# --- NEW: Pydantic Models for AI Judge Outputs ---
class ElaborationJudgement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    elaboration: float = Field(..., description="Score for encouraging a detailed and elaborate response.")

# --- NEW: AI Judge Agent Definitions ---

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

class ComprehensiveEvaluator:
    """
    Evaluates and compares question-generation systems (e.g., DCAgent vs. LLM)
    based on a ground-truth transcript, an interview outline, and pre-generated
    question suggestions.
    """

    def __init__(
        self,
        transcript_path: str,
        survey_data_path: str,
        outline_path: str,
        similarity_threshold: float = 0.5,
        verbose: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose

        # --- Load Data ---
        print("--- Loading data and models... ---")
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            self.transcript = [ConversationTurn(**turn) for turn in transcript_data["transcript"]]

            with open(survey_data_path, "r", encoding="utf-8") as f:
                self.survey_data = json.load(f)

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
        print("  [+] Loading language model for fluency...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium").eval()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gpt2.to(self.device)
        print(f"  [+] Models loaded on device: {self.device}")

        # --- Initialize Containers ---
        self.evaluation_results: List[EvaluationRound] = []

    # --- Core Calculation Methods ---

    def _cos_batch(self, main_str: str, str_list: List[str]) -> np.ndarray:
        """Calculates cosine similarity between a string and a list of strings."""
        main_str = main_str.strip()
        # Filter out empty or whitespace-only strings from the list
        str_list = [s.strip() for s in str_list if s and s.strip()]
        
        # If after cleaning, either the main string is empty or the list is empty, return zeros.
        if not main_str or not str_list:
            return np.zeros(len(str_list) if str_list else 1, dtype=np.float32)
        
        try:
            # Encode with normalization to prevent overflow
            embeddings = self.similarity_model.encode([main_str] + str_list, normalize_embeddings=True)
        except Exception as e:
            print(f"Encoding error: {e}")
            return np.zeros(len(str_list), dtype=np.float32)

        main_emb = embeddings[0]
        list_embs = embeddings[1:]

        # Since embeddings are already normalized, we can directly compute dot product
        # which equals cosine similarity for normalized vectors
        sims = np.dot(list_embs, main_emb)
        
        # Clip values to valid cosine similarity range [-1, 1]
        sims = np.clip(sims, -1.0, 1.0)
        
        # Handle any remaining NaN values
        sims = np.nan_to_num(sims, nan=0.0)
        
        return sims.astype(np.float32)

    def _fluency_score(self, text: str) -> float:
        """Calculates fluency (0-1) based on GPT-2 perplexity."""
        if not text: return 0.0
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.gpt2(tokens, labels=tokens).loss
        perplexity = math.exp(loss.item())
        # Smoothed normalization: maps perplexity to a 0-1 score.
        # Lower perplexity -> higher score. The denominator controls sensitivity.
        return math.exp(-perplexity / 100.0)

    # --- Main Evaluation Loop ---

    async def evaluate(self):
        """
        Iterates through the survey data, calculates a comprehensive set of metrics
        for each suggested question, and stores them.
        """
        print("\n=== Starting Comprehensive Evaluation ===")
        
        asked_questions_history = []
        all_time_suggestions = [] # For originality calculation

        for survey_item in self.survey_data:
            round_num = survey_item["round_num"]
            if self.verbose:
                print(f"\n[Round {round_num}] Processing...")

            # --- Establish Context for this Round ---
            prev_answer_text = survey_item["answer"].strip()
            suggestions = survey_item["suggested_questions"]
            
            # Ground truth is the next Q&A pair in the transcript
            actual_q_idx = round_num * 2
            
            actual_q_text = self.transcript[actual_q_idx].text.strip()
            if not actual_q_text:
                if self.verbose: print(f"  -> Skipping round {round_num} due to empty actual question.")
                continue

            # --- Calculate Metrics for Each Suggestion ---
            all_suggestions_this_round = [s['question'] for s in suggestions if s.get('question') and s['question'].strip()]
            if not all_suggestions_this_round:
                 if self.verbose: print(f"  -> No suggested questions for round {round_num}. Skipping.")
                 continue

            # --- NEW: Perform Per-Suggestion AI Analysis in Parallel ---
            analysis_tasks = []
            for sugg_dict in suggestions:
                q_text = sugg_dict.get('question', '').strip()
                if not q_text: continue

                # Create tasks for qualitative judge
                instruction = f"An interviewer is speaking with someone. The interviewee just said: '{prev_answer_text}'. The interviewer is considering asking the following question."
                response = q_text
                analysis_tasks.append(self._run_suggestion_analysis(instruction, response))

            analysis_results = await asyncio.gather(*analysis_tasks)
            analysis_map = {text: res for text, res in zip(all_suggestions_this_round, analysis_results)}

            qmetrics = []
            for sugg_dict in suggestions:
                q_text = sugg_dict.get('question', '').strip()
                if not q_text: continue

                # All metrics are now calculated programmatically per round
                # The single Prometheus call happens after the loop.
                # 1. Context Relevance
                ctx_rel = self._cos_batch(q_text, [prev_answer_text])[0] if prev_answer_text else 0.0

                # 2. Outline Relevance
                outline_sims = self._cos_batch(q_text, self.all_outline_questions)
                outline_rel = float(np.max(outline_sims)) if outline_sims.any() else 0.0

                # 3. Originality
                originality_sims = self._cos_batch(q_text, all_time_suggestions)
                originality = 1.0 - (float(np.max(originality_sims)) if originality_sims.any() else 0.0)

                # 5. Fluency
                flu = self._fluency_score(q_text)

                # --- NEW: Retrieve analysis results ---
                qual_res = analysis_map.get(q_text, {})

                metrics_instance = QuestionMetrics(
                    question=q_text,
                    source=sugg_dict['source'],
                    was_selected=sugg_dict['is_selected'],
                    context_relevance=float(ctx_rel),
                    outline_relevance=outline_rel,
                    originality=originality,
                )

                metrics_instance.qualitative_elaboration = qual_res.get('elaboration', 0.0)

                qmetrics.append(metrics_instance)

            selected_metric = next((m for m in qmetrics if m.was_selected), None)
            
            # We must have a selected metric from DCAgent to proceed
            if not selected_metric:
                if self.verbose: print(f"  -> No selected DCAgent question for round {round_num}. Skipping.")
                continue

            self.evaluation_results.append(EvaluationRound(
                round_num=round_num,
                actual_question=actual_q_text,
                previous_answer=prev_answer_text,
                selected_suggestion_metrics=selected_metric,
                all_suggestion_metrics=qmetrics,
            ))
            
            # Update history for the next round's originality calculation
            asked_questions_history.append(actual_q_text)
            all_time_suggestions.extend(all_suggestions_this_round)
            
        print("\n=== Evaluation Complete ===")

    async def _run_suggestion_analysis(self, instruction: str, response: str) -> Dict:
        """Runs qualitative analysis for one suggestion using an Agent."""
        try:
            # The new implementation using ElaborationJudgeAgent
            prompt = f"{instruction}\n\nPROPOSED QUESTION: {response}"
            runner_result = await Runner.run(ElaborationJudgeAgent, prompt)
            judgement = runner_result.final_output_as(ElaborationJudgement)
            if judgement:
                return {"elaboration": judgement.elaboration}
            return {"elaboration": 0.0}
        except Exception as e:
            if self.verbose:
                print(f"  -> Error during ElaborationJudgeAgent analysis: {e}")
            return {"elaboration": 0.0}

    # --- Reporting and Saving ---
    
    def _avg(self, values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
        
    def _calculate_source_stats(self, questions: List[QuestionMetrics]) -> Dict:
        if not questions: return {}
        return {
            "count": len(questions),
            "convergent_agent_selection_rate": self._avg([m.was_selected for m in questions]),
            "avg_context_relevance": self._avg([m.context_relevance for m in questions]),
            "avg_outline_relevance": self._avg([m.outline_relevance for m in questions]),
            "avg_originality": self._avg([m.originality for m in questions]),
        }

    def get_evaluation_summary(self) -> Dict:
        """Aggregates metrics across all rounds and groups them by source."""
        if not self.evaluation_results: return {}

        all_questions = [q for r in self.evaluation_results for q in r.all_suggestion_metrics]
        
        source_groups = defaultdict(list)
        for q in all_questions:
            source_key = q.source
            source_groups[source_key].append(q)

        # --- NEW: Overall Suggestion Quality ---
        all_dc_suggestions = [q for r in self.evaluation_results for q in r.all_suggestion_metrics if q.source.startswith('dcagent')]
        all_llm_suggestions = [q for r in self.evaluation_results for q in r.all_suggestion_metrics if q.source == 'llm']
        
        def get_quality_stats(questions: List[QuestionMetrics]):
            if not questions: return {}
            return {
                "max_elaboration": max(q.qualitative_elaboration for q in questions),
                "avg_elaboration": self._avg([q.qualitative_elaboration for q in questions]),
            }

        suggestion_quality_summary = {
            "dcagent_quality": get_quality_stats(all_dc_suggestions),
            "llm_quality": get_quality_stats(all_llm_suggestions)
        }

        # --- NEW: Analysis for only the questions selected by the convergent agent ---
        selected_dcagent_questions = [
            r.selected_suggestion_metrics for r in self.evaluation_results if r.selected_suggestion_metrics.source.startswith('dcagent')
        ]
        selected_llm_questions = [
            r.selected_suggestion_metrics for r in self.evaluation_results if r.selected_suggestion_metrics.source == 'llm'
        ]

        summary = {
            "total_rounds": len(self.evaluation_results),
            "by_source": {},
            "suggestion_quality_analysis": suggestion_quality_summary,
        }
        
        # --- RENAMED for clarity ---
        summary["all_suggestions_analysis"] = {
            "dcagent": {
                "avg_originality": self._avg([m.originality for m in all_dc_suggestions]),
                "avg_context_relevance": self._avg([m.context_relevance for m in all_dc_suggestions]),
                "avg_outline_relevance": self._avg([m.outline_relevance for m in all_dc_suggestions]),
                "avg_elaboration": suggestion_quality_summary["dcagent_quality"].get("avg_elaboration", 0.0),
            },
            "llm": {
                "avg_originality": self._avg([m.originality for m in all_llm_suggestions]),
                "avg_context_relevance": self._avg([m.context_relevance for m in all_llm_suggestions]),
                "avg_outline_relevance": self._avg([m.outline_relevance for m in all_llm_suggestions]),
                "avg_elaboration": suggestion_quality_summary["llm_quality"].get("avg_elaboration", 0.0),
            }
        }
        
        # --- NEW: Analysis of only the BEST selected question ---
        summary["selected_questions_analysis"] = {
            "dcagent": {
                "avg_originality": self._avg([m.originality for m in selected_dcagent_questions]),
                "avg_context_relevance": self._avg([m.context_relevance for m in selected_dcagent_questions]),
                "avg_outline_relevance": self._avg([m.outline_relevance for m in selected_dcagent_questions]),
                "avg_elaboration": self._avg([m.qualitative_elaboration for m in selected_dcagent_questions]),
            },
            "llm": {
                "avg_originality": self._avg([m.originality for m in selected_llm_questions]),
                "avg_context_relevance": self._avg([m.context_relevance for m in selected_llm_questions]),
                "avg_outline_relevance": self._avg([m.outline_relevance for m in selected_llm_questions]),
                "avg_elaboration": self._avg([m.qualitative_elaboration for m in selected_llm_questions]),
            }
        }

        # Create a combined dcagent entry for overall comparison
        dcagent_questions = [q for k, v in source_groups.items() if k.startswith('dcagent') for q in v]
        summary["by_source"]["dcagent_combined"] = self._calculate_source_stats(dcagent_questions)

        for source_name, questions in source_groups.items():
            summary["by_source"][source_name] = self._calculate_source_stats(questions)
        
        return summary

    def save_results(self, output_path: str):
        """Saves the full evaluation results and summary to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            "evaluation_summary": self.get_evaluation_summary(),
            "detailed_metrics_by_round": [asdict(r) for r in self.evaluation_results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nEvaluation results saved to: {output_path}")

async def main():
    """Main function to run the evaluation process."""
    # --- Configuration ---
    # Choose the transcript file you want to evaluate against.
    transcript_file = "mermaid_real.json"
    
    transcript_path = f"data/transcripts/{transcript_file}"
    outline_path = "data/outlines/example_outline.json"
    survey_data_path = f"data/evaluations/survey_data_v3_{transcript_file}"
    output_path = f"data/evaluations/evaluation_results_v3_{transcript_file}"

    if not os.path.exists(survey_data_path):
        print(f"Error: Survey data file not found at '{survey_data_path}'")
        print("Please run 'maji_survey_data_generator.py' first to generate it.")
        return
    
    if not os.path.exists(outline_path):
        print(f"Error: Outline file not found at '{outline_path}'")
        return

    evaluator = ComprehensiveEvaluator(
        transcript_path=transcript_path,
        survey_data_path=survey_data_path,
        outline_path=outline_path,
        verbose=True,
    )
    await evaluator.evaluate()
    evaluator.save_results(output_path)
    
    # Print summary to console for quick review
    summary = evaluator.get_evaluation_summary()
    print("\n--- Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    print("------------------------")


if __name__ == "__main__":
    asyncio.run(main()) 
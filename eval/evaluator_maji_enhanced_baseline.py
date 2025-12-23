import json
import os
import math
import re
import argparse
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
    # New Metrics
    persona_alignment: float = 0.0
    coherence: float = 0.0
    qualitative_elaboration: float = 0.0

@dataclass
class VersionRoundResult:
    """Stores metrics for one version within a single round."""
    source_name: str # e.g., 'dcagent_v2', 'llm_base'
    selected_suggestion_metrics: Optional[QuestionMetrics]
    all_suggestion_metrics: List[QuestionMetrics]
    # New insight metrics for the selected question
    insight_category: str = "SurfaceLevel"
    insight_score: float = 0.0

@dataclass
class CrossVersionEvaluationRound:
    """Stores metrics for a full round, comparing multiple versions."""
    round_num: int
    actual_question: str
    previous_answer: str
    results_by_source: Dict[str, VersionRoundResult] = field(default_factory=dict)

@dataclass
class FinalCoverageResult:
    """Stores the final outline coverage for a single source."""
    source_name: str
    coverage_percentage: float
    weighted_coverage_score: float
    covered_questions: int
    total_questions: int

# --- Pydantic Models for AI Judge Outputs ---

class ElaborationScore(BaseModel):
    model_config = ConfigDict(extra='forbid')
    elaboration: float = Field(..., description="Score for encouraging a detailed and elaborate response.")

class PersonaJudgeOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    alignment_score: float = Field(..., description="A score from 0.0 to 1.0 indicating how well the question aligns with the interviewee's persona, goals, and personality.")

class CoherenceJudgeOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    coherence_score: float = Field(..., description="A score from 0.0 to 1.0 indicating how logically and thematically connected the proposed question is to the preceding exchange.")

class InsightJudgement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    insight_category: str = Field(..., description="The category of insight. Must be one of: 'Connecting', 'Challenging', 'Motivational', 'Hypothetical', 'SurfaceLevel'.")
    reasoning: str = Field(..., description="A brief justification for the chosen category.")

class PlanEvaluationOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plan_relevance: float = Field(..., description="Score from 0.0-1.0 for how relevant the agent plan is to the immediate conversational context.")
    plan_creativity: float = Field(..., description="Score from 0.0-1.0 for how creative and non-obvious the agent plan is.")

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
    output_type=ElaborationScore,
)

PersonaJudgeAgent = Agent(
    name="PersonaJudgeAgent",
    instructions=(
        "You are an expert profiler and interviewer. You will be given an interviewee's persona and a proposed question. "
        "Your task is to evaluate how well the question aligns with the interviewee's stated background, personality, and goals. "
        "A high score means the question would be engaging, relevant, and interesting *to this specific person*. "
        "A low score means it is too generic, irrelevant, or misaligned with their character.\n"
        "Your output **MUST** be a single JSON object with the key: `alignment_score` (a float from 0.0 to 1.0)."
    ),
    model="gpt-4o-mini",
    output_type=PersonaJudgeOutput,
)

CoherenceJudgeAgent = Agent(
    name="CoherenceJudgeAgent",
    instructions=(
        "You are an expert in discourse analysis. You will be given the last question asked, the answer given, and a new proposed question. "
        "Your task is to evaluate the logical and thematic coherence of the new question as a follow-up. "
        "A high score means the question is a sensible, well-connected continuation of the dialogue. "
        "A low score means it feels abrupt, random, disconnected, or ignores the context of the previous answer.\n"
        "Your output **MUST** be a single JSON object with the key: `coherence_score` (a float from 0.0 to 1.0)."
    ),
    model="gpt-4o-mini",
    output_type=CoherenceJudgeOutput,
)

PlanEvaluatorAgent = Agent(
    name="PlanEvaluatorAgent",
    instructions=(
        "You are an expert evaluator of AI agent systems. Your task is to assess the quality of a *plan* for generating interview questions, not the questions themselves. "
        "The plan consists of a list of specialist agents that will be created to handle the current situation. "
        "Evaluate the plan based on the conversational context.\n\n"
        "1.  **Plan Relevance**: How well does the chosen set of agents address the immediate needs of the conversation? (e.g., if the user is being emotional, is there an 'Emotion' agent planned?).\n"
        "2.  **Plan Creativity**: How creative is the plan? Does it propose novel specialists to find unique angles, or is it a generic, boilerplate plan?\n\n"
        "Your output **MUST** be a single JSON object with the keys: `plan_relevance` and `plan_creativity`."
    ),
    model="gpt-4o-mini",
    output_type=PlanEvaluationOutput,
)

InsightJudgeAgent = Agent(
    name="InsightJudgeAgent",
    instructions=(
        "You are an expert conversation analyst. Your task is to categorize a proposed interview question based on the **full context of the interview history**. "
        "Analyze how the question relates to the entire dialogue, not just the last turn.\n\n"
        "Categories:\n"
        "- `Connecting`: The question links the current topic to a **significantly earlier** part of the conversation (more than 2-3 turns ago).\n"
        "- `Challenging`: The question identifies and probes a potential contradiction, inconsistency, or assumption in the interviewee's statements.\n"
        "- `Motivational`: The question explores the deep-seated 'why' behind an answer, focusing on core values, goals, or driving forces.\n"
        "- `Hypothetical`: The question poses a creative 'what if' scenario to explore the interviewee's principles or thinking process.\n"
        "- `SurfaceLevel`: A standard, logical follow-up that explores the immediate topic but lacks a deeper connection or creative angle.\n\n"
        "Your output **MUST** be a single JSON object with the keys: `insight_category` and `reasoning`."
    ),
    model="gpt-4o-mini",
    output_type=InsightJudgement,
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
        comparison_data_path: str, # A single path to the combined comparison data
        outline_path: str,
        persona_path: str,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.persona_str = self._format_persona(persona_path)
        
        # --- Load Data ---
        print("--- Loading data and models... ---")
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            self.transcript = [ConversationTurn(**turn) for turn in transcript_data["transcript"]]

            with open(comparison_data_path, "r", encoding="utf-8") as f:
                self.comparison_data = json.load(f)

            with open(outline_path, "r", encoding="utf-8") as f:
                outline_json = json.load(f)
            self.outline = [OutlineSection(**s) for s in outline_json]
            self.all_outline_questions = [q.question for s in self.outline for q in s.questions]
            self.outline_importance_map = {q.question: q.importance for s in self.outline for q in s.questions}

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
        self.final_coverage_results: Dict[str, FinalCoverageResult] = {}
        self._initialize_coverage_trackers()

    def _format_persona(self, persona_path: str) -> str:
        """Loads and formats the persona for prompts."""
        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona_data = json.load(f)['persona']
            parts = [
                f"Name: {persona_data.get('name', 'N/A')}",
                f"Personality: {persona_data.get('personality', 'N/A')}",
                f"Background: {persona_data.get('background', 'N/A')}"
            ]
            return "\n".join(parts)
        except Exception as e:
            print(f"Warning: Could not load or format persona from {persona_path}. Details: {e}")
            return "No persona information available."

    def _initialize_coverage_trackers(self):
        """Initializes coverage trackers for all potential sources."""
        self.all_sources = set()
        for item in self.comparison_data:
            for s in item['suggested_questions']:
                base_source = '_'.join(s['source'].split('_')[:2]) if s['source'].startswith('dcagent') else s['source']
                self.all_sources.add(base_source)

        self.coverage_trackers = {
            source: {s.id: {q.id: False for q in s.questions} for s in self.outline}
            for source in self.all_sources
        }

    def _clean_question_text(self, text: str) -> str:
        """Cleans up raw text from LLM to extract a valid question."""
        if not isinstance(text, str):
            return ""
        text = text.strip()

        # Remove common conversational artifacts and code blocks
        text = re.sub(r'```json\s*', '', text, flags=re.I)
        text = re.sub(r'```', '', text, flags=re.I)
        text = re.sub(r'Step-by-step reasoning:', '', text, flags=re.I)
        text = re.sub(r'Final output:', '', text, flags=re.I)
        text = re.sub(r'Path \d+.*', '', text, flags=re.I)
        text = re.sub(r'Synthesis:', '', text, flags=re.I)

        # Try to parse as JSON if it looks like it, and take the first element if it's a list
        if text.startswith('[') and text.endswith(']'):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0]).strip().strip('"')
            except json.JSONDecodeError:
                pass # Not valid JSON, proceed with string cleaning

        # Take the first non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return ""

        # Remove leading list-like characters and quotes
        first_line = lines[0]
        first_line = re.sub(r'^\s*-\s*', '', first_line)
        first_line = re.sub(r'^\s*\d+\.\s*', '', first_line)
        first_line = first_line.strip().strip('"').strip("'").strip("`")

        return first_line

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

            # Find the corresponding round in the single comparison data file
            survey_item = next((item for item in self.comparison_data if item['round_num'] == round_num), None)
            if not survey_item:
                if self.verbose: print(f"  -> No survey data found for round {round_num}. Skipping.")
                continue

            # --- V3 Plan Evaluation (if applicable) ---
            v3_plan = None
            v3_plan_scores = {}
            # Find the plan attached to any v3 suggestion in this round
            for s in survey_item['suggested_questions']:
                if s.get('agent_specific_data') and 'divergent_plan' in s['agent_specific_data']:
                    v3_plan = s['agent_specific_data']['divergent_plan']
                    break
            
            if v3_plan:
                plan_eval_input = (
                    f"INTERVIEW CONTEXT:\nPrevious Question: {actual_q_text}\nPrevious Answer: {prev_answer_text}\n\n"
                    f"AGENT PLAN TO EVALUATE:\n{json.dumps(v3_plan, indent=2)}\n\n"
                    "Please evaluate this plan."
                )
                try:
                    plan_res = await Runner.run(PlanEvaluatorAgent, plan_eval_input)
                    plan_output = plan_res.final_output_as(PlanEvaluationOutput)
                    if plan_output:
                        v3_plan_scores = plan_output.model_dump()
                        if self.verbose: print(f"  [+] V3 Plan evaluated: {v3_plan_scores}")
                except Exception as e:
                    if self.verbose: print(f"  -> PlanEvaluatorAgent failed: {e}")
            # --- End of Plan Evaluation ---

            # Group suggestions by their source (e.g., 'dcagent_v1_...', 'llm_base')
            suggestions_by_source = defaultdict(list)
            for s in survey_item['suggested_questions']:
                # Simplify source name, e.g., 'dcagent_v1_someAgent' -> 'dcagent_v1'
                base_source = '_'.join(s['source'].split('_')[:2]) if s['source'].startswith('dcagent') else s['source']
                suggestions_by_source[base_source].append(s)

            # Always filter to only V1 if it exists (user only wants V1 results)
            if 'dcagent_v1' in suggestions_by_source:
                sources_to_process = ['dcagent_v1']
            else:
                # If V1 doesn't exist, process all (backward compatibility)
                sources_to_process = list(suggestions_by_source.keys())

            for source_name in sources_to_process:
                source_suggestions = suggestions_by_source[source_name]
                if not source_suggestions:
                    continue

                # --- Clean the question text before processing ---
                cleaned_suggestions = []
                for s in source_suggestions:
                    # Create a copy to avoid modifying the original data structure
                    s_copy = s.copy()
                    s_copy['question'] = self._clean_question_text(s['question'])
                    if s_copy['question']: # Only add if a valid question remains
                        cleaned_suggestions.append(s_copy)
                
                if not cleaned_suggestions:
                    continue
                # --- End of Cleaning ---

                suggestion_texts = [s['question'] for s in cleaned_suggestions]

                # --- Perform Per-Suggestion AI Analysis in Parallel ---
                analysis_tasks = []
                qualitative_instruction = f"An interviewer is speaking with someone. The interviewee just said: '{prev_answer_text}'. The interviewer is considering asking the following question."
                persona_instruction = f"INTERVIEWEE PERSONA:\n{self.persona_str}\n\nBased on this persona, evaluate the following proposed question."
                coherence_instruction = f"PREVIOUS QUESTION:\n{actual_q_text}\n\nPREVIOUS ANSWER:\n{prev_answer_text}\n\nBased on this exchange, evaluate the coherence of the following PROPOSED QUESTION."

                for q_text in suggestion_texts:
                    analysis_tasks.append(self._run_full_suggestion_analysis(
                        qualitative_instruction, persona_instruction, coherence_instruction, q_text
                    ))
                
                analysis_results = await asyncio.gather(*analysis_tasks)
                analysis_map = {text: res for text, res in zip(suggestion_texts, analysis_results)}

                # --- Calculate Metrics for each DCAgent suggestion ---
                qmetrics = []
                for sugg_dict in cleaned_suggestions:
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
                        persona_alignment=qual_res.get('persona_alignment', 0.0),
                        coherence=qual_res.get('coherence', 0.0),
                        qualitative_elaboration=qual_res.get('elaboration', 0.0),
                    ))

                if not qmetrics: continue

                selected_metric = next((m for m in qmetrics if m.was_selected), None)
                
                # --- Calculate Option Set Richness ---
                option_set_richness = 0.0
                agent_data = source_suggestions[0].get('agent_specific_data')
                if agent_data and 'option_set' in agent_data:
                    option_set = agent_data['option_set']
                    # Ensure questions are strings
                    if option_set and isinstance(option_set[0], dict):
                        q_texts = [q['question'] for q in option_set]
                    else:
                        q_texts = [str(q) for q in option_set]
                    
                    if len(q_texts) > 1:
                        embeddings = self.similarity_model.encode(q_texts, normalize_embeddings=True)
                        # Calculate average pairwise cosine distance (1 - similarity)
                        distances = 1 - np.dot(embeddings, embeddings.T)
                        # Sum of upper triangle of the distance matrix, divided by number of pairs
                        num_pairs = len(q_texts) * (len(q_texts) - 1) / 2
                        option_set_richness = float(np.sum(np.triu(distances, k=1)) / num_pairs)

                round_evaluation.results_by_source[source_name] = VersionRoundResult(
                    source_name=source_name,
                    selected_suggestion_metrics=selected_metric,
                    all_suggestion_metrics=qmetrics,
                    # New insight metrics for the selected question
                    insight_category="SurfaceLevel",
                    insight_score=0.0,
                )
                # Attach Option Set Richness score to the result for this round
                setattr(round_evaluation.results_by_source[source_name], 'option_set_richness', option_set_richness)

                # Attach winning specialist to V2/V3 results
                if source_name in ['dcagent_v2', 'dcagent_v3'] and agent_data and 'winning_specialist' in agent_data:
                    setattr(round_evaluation.results_by_source[source_name], 'winning_specialist', agent_data['winning_specialist'])

                # --- Update Coverage Tracker ---
                if selected_metric:
                    # Simple heuristic: find the closest matching outline question to the *selected* question
                    # In a real scenario, this would come from the agent's internal state.
                    similarities = self._cos_batch(selected_metric.question, self.all_outline_questions)
                    if np.max(similarities) > 0.7: # Confidence threshold
                        best_match_idx = np.argmax(similarities)
                        matched_q_text = self.all_outline_questions[best_match_idx]
                        # Find the section and question ID for this text
                        for s in self.outline:
                            for q in s.questions:
                                if q.question == matched_q_text:
                                    self.coverage_trackers[source_name][s.id][q.id] = True
                                    if self.verbose:
                                        print(f"  -> Marked QID {s.id}{q.id} as covered for {source_name}")
                                    break
                # --- End of Coverage Update ---

            if round_evaluation.results_by_source:
                self.evaluation_results.append(round_evaluation)
                all_time_suggestions.extend([
                    q.question for res in round_evaluation.results_by_source.values() for q in res.all_suggestion_metrics
                ])

            # --- Insight Analysis for Selected Questions ---
            transcript_history_str = "\n".join([f"Q: {self.transcript[i].text}\nA: {self.transcript[i+1].text}" for i in range(0, actual_q_idx, 2)])
            
            for source_name, result in round_evaluation.results_by_source.items():
                if result.selected_suggestion_metrics:
                    selected_question = result.selected_suggestion_metrics.question
                    insight_prompt = (
                        f"INTERVIEW HISTORY:\n{transcript_history_str}\n\n"
                        f"PREVIOUS QUESTION: {actual_q_text}\n\n"
                        f"PREVIOUS ANSWER: {prev_answer_text}\n\n"
                        f"PROPOSED NEXT QUESTION TO CATEGORIZE:\n'{selected_question}'"
                    )
                    try:
                        insight_res = await Runner.run(InsightJudgeAgent, insight_prompt)
                        insight_output = insight_res.final_output_as(InsightJudgement)
                        if insight_output:
                            result.insight_category = insight_output.insight_category
                            # Assign a numerical score for trajectory analysis
                            result.insight_score = {"Connecting": 4, "Challenging": 4, "Motivational": 3, "Hypothetical": 2, "SurfaceLevel": 1}.get(insight_output.insight_category, 0)
                            if self.verbose: print(f"  [Insight - {source_name}]: {insight_output.insight_category}")
                    except Exception as e:
                        if self.verbose: print(f"  -> InsightJudgeAgent failed for {source_name}: {e}")
            # --- End Insight Analysis ---

            # Attach plan scores to the V3 result for this round
            if v3_plan_scores and 'dcagent_v3' in round_evaluation.results_by_source:
                # This is a bit of a hack, but we'll store it on the round result for now.
                round_evaluation.results_by_source['dcagent_v3'].plan_scores = v3_plan_scores

        self._calculate_final_coverage()
        print("\n=== Evaluation Complete ===")

    def _calculate_final_coverage(self):
        """Calculates and stores the final coverage percentage for each source."""
        print("\n--- Calculating Final Outline Coverage ---")
        
        importance_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        total_possible_weighted_score = sum(importance_weights.get(q.importance.lower(), 1) for s in self.outline for q in s.questions)
        total_questions = len(self.all_outline_questions)

        if total_questions == 0: return

        for source_name, tracker in self.coverage_trackers.items():
            covered_count = 0
            current_weighted_score = 0
            for section in self.outline:
                for question in section.questions:
                    if tracker[section.id][question.id]:
                        covered_count += 1
                        current_weighted_score += importance_weights.get(question.importance.lower(), 1)
            
            coverage_percentage = (covered_count / total_questions) * 100 if total_questions > 0 else 0
            weighted_score_percentage = (current_weighted_score / total_possible_weighted_score) * 100 if total_possible_weighted_score > 0 else 0

            self.final_coverage_results[source_name] = FinalCoverageResult(
                source_name=source_name,
                coverage_percentage=coverage_percentage,
                weighted_coverage_score=weighted_score_percentage,
                covered_questions=covered_count,
                total_questions=total_questions,
            )
            print(f"  - {source_name}: Coverage={coverage_percentage:.2f}%, Weighted Score={weighted_score_percentage:.2f}%")

    async def _run_full_suggestion_analysis(
        self, qualitative_instruction: str, persona_instruction: str, coherence_instruction: str, response: str
    ) -> Dict:
        """Runs all qualitative judge agents in parallel for a single suggestion."""
        qual_prompt = f"{qualitative_instruction}\n\nPROPOSED QUESTION: {response}"
        pers_prompt = f"{persona_instruction}\n\nPROPOSED QUESTION: {response}"
        coher_prompt = f"{coherence_instruction}\n\nPROPOSED QUESTION: {response}"

        tasks = {
            "elaboration": Runner.run(ElaborationJudgeAgent, qual_prompt),
            "persona": Runner.run(PersonaJudgeAgent, pers_prompt),
            "coherence": Runner.run(CoherenceJudgeAgent, coher_prompt),
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        final_scores = {}
        task_keys = list(tasks.keys())

        # Process qualitative results
        if not isinstance(results[0], Exception) and results[0].final_output_as(ElaborationScore):
            final_scores.update(results[0].final_output_as(ElaborationScore).model_dump())
        else:
            if self.verbose: print(f"  -> Error during ElaborationJudgeAgent analysis: {results[0]}")
        
        # Process persona results
        if not isinstance(results[1], Exception) and results[1].final_output_as(PersonaJudgeOutput):
            final_scores['persona_alignment'] = results[1].final_output_as(PersonaJudgeOutput).alignment_score
        else:
             if self.verbose: print(f"  -> Error during PersonaJudgeAgent analysis: {results[1]}")

        # Process coherence results
        if not isinstance(results[2], Exception) and results[2].final_output_as(CoherenceJudgeOutput):
            final_scores['coherence'] = results[2].final_output_as(CoherenceJudgeOutput).coherence_score
        else:
            if self.verbose: print(f"  -> Error during CoherenceJudgeAgent analysis: {results[2]}")

        return final_scores

    # --- Reporting and Saving ---
    
    def _avg(self, values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
        
    def get_evaluation_summary(self) -> Dict:
        all_sources = sorted(list(set(
            source for r in self.evaluation_results for source in r.results_by_source.keys()
        )))
        summary = {source: {} for source in all_sources}

        for source in all_sources:
            all_suggestions = [q for r in self.evaluation_results for q in r.results_by_source.get(source, VersionRoundResult(source, None, [], 0)).all_suggestion_metrics]
            selected_suggestions = [r.results_by_source.get(source, VersionRoundResult(source, None, [], 0)).selected_suggestion_metrics for r in self.evaluation_results if r.results_by_source.get(source, VersionRoundResult(source, None, [], 0)).selected_suggestion_metrics]
            
            # --- New Metric Calculations ---
            # 1. Source Diversity for selected questions
            selected_sources = [
                m.source for m in selected_suggestions 
                if m.source.startswith('dcagent_v2') or m.source.startswith('dcagent_v3')
            ]
            unique_divergent_agents = set(s.split('_')[-1] for s in selected_sources)
            source_diversity = len(unique_divergent_agents) / len(selected_sources) if selected_sources else 0

            # 2. V3 Plan Quality
            plan_scores = [
                getattr(r.results_by_source.get('dcagent_v3'), 'plan_scores', {})
                for r in self.evaluation_results
                if 'dcagent_v3' in r.results_by_source and hasattr(r.results_by_source['dcagent_v3'], 'plan_scores')
            ]
            
            avg_plan_scores = {}
            if plan_scores:
                avg_plan_scores = {
                    f"avg_{key}": self._avg([s.get(key, 0.0) for s in plan_scores])
                    for key in plan_scores[0].keys()
                }

            # 3. Option Set Richness
            avg_option_set_richness = self._avg([
                getattr(res, 'option_set_richness', 0.0)
                for r in self.evaluation_results for res in r.results_by_source.values()
                if res.source_name == source
            ])

            # 4. Specialist Win Rate (V2/V3 only)
            specialist_wins = defaultdict(int)
            total_wins = 0
            if source in ['dcagent_v2', 'dcagent_v3']:
                for r in self.evaluation_results:
                    source_result = r.results_by_source.get(source)
                    if source_result and hasattr(source_result, 'winning_specialist'):
                        specialist_name = source_result.winning_specialist.replace(f'dcagent_{source}_', '')
                        specialist_wins[specialist_name] += 1
                        total_wins += 1
            
            specialist_win_rates = {f"win_rate_{k}": v / total_wins for k, v in specialist_wins.items()} if total_wins > 0 else {}

            # 5. Insight Profile & Trajectory
            num_rounds = len(self.evaluation_results)
            first_half_rounds = num_rounds // 2
            
            insight_categories = defaultdict(int)
            first_half_scores, second_half_scores = [], []

            for i, r in enumerate(self.evaluation_results):
                source_result = r.results_by_source.get(source)
                if source_result:
                    insight_categories[source_result.insight_category] += 1
                    if i < first_half_rounds:
                        first_half_scores.append(source_result.insight_score)
                    else:
                        second_half_scores.append(source_result.insight_score)
            
            total_insights = sum(insight_categories.values())
            insight_profile = {f"profile_{k}": v / total_insights for k, v in insight_categories.items()} if total_insights > 0 else {}

            avg_first_half = self._avg(first_half_scores)
            avg_second_half = self._avg(second_half_scores)
            insight_improvement = ((avg_second_half - avg_first_half) / avg_first_half * 100) if avg_first_half > 0 else 0.0

            insight_trajectory = {
                "avg_insight_score_first_half": avg_first_half,
                "avg_insight_score_second_half": avg_second_half,
                "insight_improvement_rate_pct": insight_improvement
            }

            def get_quality_stats(questions: List[QuestionMetrics]):
                if not questions: return {}
                return {
                    "avg_elaboration": self._avg([q.qualitative_elaboration for q in questions]),
                    "avg_originality": self._avg([m.originality for m in questions]),
                    "avg_context_relevance": self._avg([m.context_relevance for m in questions]),
                    "avg_outline_relevance": self._avg([m.outline_relevance for m in questions]),
                    "avg_persona_alignment": self._avg([q.persona_alignment for q in questions]),
                    "avg_coherence": self._avg([q.coherence for q in questions]),
                }
            
            summary[source]["all_suggestions_analysis"] = get_quality_stats(all_suggestions)
            summary[source]["selected_questions_analysis"] = get_quality_stats(selected_suggestions)
            summary[source]["strategic_analysis"] = {
                "source_diversity": source_diversity,
                "option_set_richness": avg_option_set_richness,
                **specialist_win_rates,
                **insight_profile,
                **insight_trajectory
            }
            # Add V3 plan scores only to the V3 summary
            if source == 'dcagent_v3' and avg_plan_scores:
                summary[source]["strategic_analysis"].update(avg_plan_scores)

        return summary

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = {
            "evaluation_summary": self.get_evaluation_summary(),
            "final_coverage": {k: asdict(v) for k, v in self.final_coverage_results.items()},
            "detailed_metrics_by_round": [asdict(r) for r in self.evaluation_results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nEvaluation results saved to: {output_path}")

async def main():
    """Main function to run the cross-version evaluation process."""
    parser = argparse.ArgumentParser(description='Run cross-version evaluation on MAJI comparison data.')
    parser.add_argument('--transcript-path', type=str, default='data/transcripts/mermaid_real.json',
                        help='Path to the input transcript JSON file.')
    parser.add_argument('--outline-path', type=str, default='data/outlines/example_outline.json',
                        help='Path to the outline JSON file.')
    parser.add_argument('--persona-path', type=str, default='data/personas/example_persona.json', help='Path to the persona JSON file.')
    args = parser.parse_args()

    transcript_file = os.path.basename(args.transcript_path)
    
    transcript_path = args.transcript_path
    outline_path = args.outline_path
    
    # Path to the single, enhanced comparison data file
    comparison_data_path = f"data/evaluations/comparison_data_enhanced_{transcript_file}"
    output_path = f"data/evaluations/evaluation_results_enhanced_{transcript_file}"

    if not os.path.exists(comparison_data_path):
        print(f"Error: Comparison data file not found at '{comparison_data_path}'")
        print("Please run the comparison data generator first.")
        return

    evaluator = MajiEvaluator(
        transcript_path=transcript_path,
        comparison_data_path=comparison_data_path,
        outline_path=outline_path,
        persona_path=args.persona_path,
        verbose=True,
    )
    await evaluator.evaluate()
    evaluator.save_results(output_path)
    
    summary = evaluator.get_evaluation_summary()
    print("\n--- Cross-Version MAJI Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    print("-----------------------------------------")

    print("\n--- Final Outline Coverage Summary ---")
    print(json.dumps({k: asdict(v) for k, v in evaluator.final_coverage_results.items()}, indent=2))
    print("------------------------------------")

if __name__ == "__main__":
    asyncio.run(main()) 
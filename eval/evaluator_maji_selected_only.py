"""
Modified version of evaluator_maji_enhanced_baseline.py that only evaluates selected questions.
This script evaluates only the questions that were actually selected by each agent.
"""
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

# Import from the original evaluator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator_maji_enhanced_baseline import (
    ConversationTurn, OutlineQuestion, OutlineSection,
    QuestionMetrics, VersionRoundResult, CrossVersionEvaluationRound,
    FinalCoverageResult, ElaborationScore, PersonaJudgeOutput,
    CoherenceJudgeOutput, InsightJudgement, ElaborationJudgeAgent,
    PersonaJudgeAgent, CoherenceJudgeAgent, InsightJudgeAgent
)

# --- Environment and Client Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# --- Main Evaluator Class (Modified) ---

class MajiEvaluatorSelectedOnly:
    """
    Evaluates only the selected questions from different versions of the DCAgent system.
    This is a modified version that filters to only evaluate selected questions.
    """

    def __init__(
        self,
        transcript_path: str,
        comparison_data_path: str,
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

    # --- Main Evaluation Loop (Modified to only evaluate selected questions) ---

    async def evaluate(self):
        print("\n=== Starting Cross-Version MAJI Evaluation (Selected Questions Only) ===")
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

                # --- Filter to only selected questions ---
                selected_suggestions = [s for s in source_suggestions if s.get('is_selected', False)]
                
                if not selected_suggestions:
                    if self.verbose: print(f"  -> No selected question for {source_name} in round {round_num}. Skipping.")
                    continue

                # Clean the selected question text
                cleaned_selected = []
                for s in selected_suggestions:
                    s_copy = s.copy()
                    s_copy['question'] = self._clean_question_text(s['question'])
                    if s_copy['question']:
                        cleaned_selected.append(s_copy)
                
                if not cleaned_selected:
                    continue

                # Only evaluate the first selected question (should typically be just one)
                selected_s = cleaned_selected[0]
                selected_question_text = selected_s['question']

                # --- Perform AI Analysis on selected question ---
                qualitative_instruction = f"An interviewer is speaking with someone. The interviewee just said: '{prev_answer_text}'. The interviewer is considering asking the following question."
                persona_instruction = f"INTERVIEWEE PERSONA:\n{self.persona_str}\n\nBased on this persona, evaluate the following proposed question."
                coherence_instruction = f"PREVIOUS QUESTION:\n{actual_q_text}\n\nPREVIOUS ANSWER:\n{prev_answer_text}\n\nBased on this exchange, evaluate the coherence of the following PROPOSED QUESTION."

                qual_res = await self._run_full_suggestion_analysis(
                    qualitative_instruction, persona_instruction, coherence_instruction, selected_question_text
                )

                # Calculate metrics for selected question
                ctx_rel = self._cos_batch(selected_question_text, [prev_answer_text])[0] if prev_answer_text else 0.0
                outline_rel = float(np.max(self._cos_batch(selected_question_text, self.all_outline_questions)))
                originality = 1.0 - (float(np.max(self._cos_batch(selected_question_text, all_time_suggestions))) if all_time_suggestions else 0.0)
                
                selected_metric = QuestionMetrics(
                    question=selected_question_text,
                    source=selected_s['source'],
                    was_selected=True,
                    context_relevance=float(ctx_rel),
                    outline_relevance=outline_rel,
                    originality=originality,
                    persona_alignment=qual_res.get('persona_alignment', 0.0),
                    coherence=qual_res.get('coherence', 0.0),
                    qualitative_elaboration=qual_res.get('elaboration', 0.0),
                )

                # --- Insight Analysis for Selected Question ---
                transcript_history_str = "\n".join([f"Q: {self.transcript[i].text}\nA: {self.transcript[i+1].text}" for i in range(0, actual_q_idx, 2)])
                insight_prompt = (
                    f"INTERVIEW HISTORY:\n{transcript_history_str}\n\n"
                    f"PREVIOUS QUESTION: {actual_q_text}\n\n"
                    f"PREVIOUS ANSWER: {prev_answer_text}\n\n"
                    f"PROPOSED NEXT QUESTION TO CATEGORIZE:\n'{selected_question_text}'"
                )
                insight_category = "SurfaceLevel"
                insight_score = 0.0
                try:
                    insight_res = await Runner.run(InsightJudgeAgent, insight_prompt)
                    insight_output = insight_res.final_output_as(InsightJudgement)
                    if insight_output:
                        insight_category = insight_output.insight_category
                        insight_score = {"Connecting": 4, "Challenging": 4, "Motivational": 3, "Hypothetical": 2, "SurfaceLevel": 1}.get(insight_output.insight_category, 0)
                        if self.verbose: print(f"  [Insight - {source_name}]: {insight_category}")
                except Exception as e:
                    if self.verbose: print(f"  -> InsightJudgeAgent failed for {source_name}: {e}")

                round_evaluation.results_by_source[source_name] = VersionRoundResult(
                    source_name=source_name,
                    selected_suggestion_metrics=selected_metric,
                    all_suggestion_metrics=[selected_metric],  # Only selected question
                    insight_category=insight_category,
                    insight_score=insight_score,
                )

                # --- Update Coverage Tracker ---
                if selected_metric:
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

                # Add to history for originality calculation
                all_time_suggestions.append(selected_question_text)

            if round_evaluation.results_by_source:
                self.evaluation_results.append(round_evaluation)

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
            selected_suggestions = [r.results_by_source.get(source, VersionRoundResult(source, None, [], 0)).selected_suggestion_metrics for r in self.evaluation_results if r.results_by_source.get(source, VersionRoundResult(source, None, [], 0)).selected_suggestion_metrics]
            
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
            
            summary[source]["selected_questions_analysis"] = get_quality_stats(selected_suggestions)

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
    """Main function to run the cross-version evaluation process (selected questions only)."""
    parser = argparse.ArgumentParser(description='Run cross-version evaluation on MAJI comparison data (selected questions only).')
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
    output_path = f"data/evaluations/evaluation_results_enhanced_selected_{transcript_file}"

    if not os.path.exists(comparison_data_path):
        print(f"Error: Comparison data file not found at '{comparison_data_path}'")
        print("Please run the comparison data generator first.")
        return

    evaluator = MajiEvaluatorSelectedOnly(
        transcript_path=transcript_path,
        comparison_data_path=comparison_data_path,
        outline_path=outline_path,
        persona_path=args.persona_path,
        verbose=True,
    )
    await evaluator.evaluate()
    evaluator.save_results(output_path)
    
    summary = evaluator.get_evaluation_summary()
    print("\n--- Cross-Version MAJI Evaluation Summary (Selected Questions Only) ---")
    print(json.dumps(summary, indent=2))
    print("-----------------------------------------")

    print("\n--- Final Outline Coverage Summary ---")
    print(json.dumps({k: asdict(v) for k, v in evaluator.final_coverage_results.items()}, indent=2))
    print("------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())


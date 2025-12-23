"""
Modified version of evaluator_prometheus_enhanced.py that only evaluates selected questions.
This script evaluates only the questions that were actually selected by each agent.
"""
import json
import os
import asyncio
import argparse
import itertools
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import warnings
import dataclasses
from tqdm import tqdm

# Import from the original Prometheus evaluator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator_prometheus_enhanced import (
    JUDGE_MODEL, MODELS_TO_EVALUATE, ALL_METRIC_RUBRICS,
    QuestionMetrics, ModelRoundResult, EvaluationRound
)

# --- Prometheus Imports ---
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

warnings.filterwarnings("ignore", category=UserWarning)

# --- MAIN EVALUATOR CLASS (Modified) ---
class PrometheusEvaluatorSelectedOnly:
    def __init__(self, comparison_data_path: str, transcript_path: str, outline_path: str, persona_path: str, verbose: bool = True, num_rounds: Optional[int] = None, judge_model: Optional[str] = None):
        self.verbose = verbose
        self.model_names = MODELS_TO_EVALUATE
        self.num_rounds = num_rounds
        self.comparison_data_path = comparison_data_path
        
        print("--- Loading data... ---")
        try:
            with open(comparison_data_path, "r", encoding="utf-8") as f:
                self.comparison_data = json.load(f)
            with open(transcript_path, "r", encoding="utf-8") as f:
                self.transcript = json.load(f)["transcript"]
            with open(outline_path, "r", encoding="utf-8") as f:
                self.outline = json.load(f)
            with open(persona_path, "r", encoding="utf-8") as f:
                self.persona = json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

        # Use provided judge_model, or fall back to environment variable or default
        self.judge_model = judge_model or os.getenv("PROMETHEUS_MODEL_PATH", JUDGE_MODEL)
        print(f"--- Initializing Prometheus Judge ({self.judge_model}) ---")
        
        # Check if model path exists (if it's a local path)
        # HuggingFace model IDs typically contain "/" or "@", URLs contain "http", so skip check for those
        is_hf_id = "/" in self.judge_model or "@" in self.judge_model
        is_url = self.judge_model.startswith("http")
        is_local = not is_hf_id and not is_url
        
        if is_local and not os.path.exists(self.judge_model):
            raise ValueError(
                f"Prometheus model not found at '{self.judge_model}'. "
                f"Please either:\n"
                f"1. Set PROMETHEUS_MODEL_PATH environment variable to the model path or HuggingFace ID\n"
                f"2. Pass --judge-model argument with the model path or HuggingFace ID (e.g., 'prometheus-eval/prometheus-7b-v2')\n"
                f"3. Download the model locally and specify the path"
            )
        
        self.model = VLLM(model=self.judge_model)
        self.judge = PrometheusEval(model=self.model, absolute_grade_template=ABSOLUTE_PROMPT, relative_grade_template=RELATIVE_PROMPT)
        self.evaluation_results: List[EvaluationRound] = []

    def _get_source_prefix(self, model_name: str) -> str:
        """Gets the expected source prefix string for a given model name."""
        return f'dcagent_{model_name}' if model_name in ['v1', 'v2', 'v3'] else model_name

    def _get_selected_question_for_source(self, suggestions: List[Dict], source_prefix: str) -> Optional[str]:
        """Gets the selected question for a given source (only returns selected questions)."""
        for s in suggestions:
            if s.get('source', '').startswith(source_prefix) and s.get('is_selected', False):
                return s.get('question', '')
        return None

    async def _get_absolute_scores(self, question: str, prev_answer: str, per_model_q_history: List[str], full_convo_history: List[Dict]) -> QuestionMetrics:
        """Gets all metric scores for a single question."""
        scores = {}
        feedbacks = {}
        
        # Prepare context strings
        persona_str = f"Interviewer Persona:\n{json.dumps(self.persona, indent=2)}"
        outline_str = f"Interview Outline:\n{json.dumps(self.outline, indent=2)}\n\nPrevious Turn:\nInterviewer: ...\nInterviewee: {prev_answer}"
        synthesis_str = f"Full Conversation History (Question-Answer pairs):\n{json.dumps(full_convo_history, indent=2)}\n\nPrevious Turn:\nInterviewer: ...\nInterviewee: {prev_answer}"
        originality_str = f"History of Previously Suggested Questions for this model:\n{json.dumps(per_model_q_history, indent=2)}"

        for name, rubric in ALL_METRIC_RUBRICS.items():
            instruction = prev_answer # Default for context_relevance, insight, coherence, qualitative_elaboration
            if name == "originality":
                instruction = originality_str
            elif name == "persona_alignment":
                instruction = persona_str
            elif name == "outline_relevance":
                instruction = outline_str
            elif name == "conversational_synthesis":
                instruction = synthesis_str
            
            try:
                feedback, score = await asyncio.to_thread(
                    self.judge.single_absolute_grade,
                    instruction=instruction, response=question, rubric=rubric, reference_answer=""
                )
                scores[name] = (score - 1) / 4.0 if score and score > 0 else 0.0
                feedbacks[name] = feedback
            except Exception as e:
                if self.verbose: print(f"  -> Prometheus error for {name}: {e}")
                scores[name], feedbacks[name] = 0.0, f"Error: {e}"
        return QuestionMetrics(question=question, scores=scores, feedback=feedbacks)

    async def evaluate(self):
        """Processes the entire comparison file round by round (selected questions only)."""
        file_name = os.path.basename(self.comparison_data_path)
        print(f"\n=== Starting Prometheus Evaluation for Selected Questions Only: {file_name} ===")
        all_time_suggestions = defaultdict(list)
        conversation_history = []

        num_rounds_to_process = len(self.comparison_data)
        if self.num_rounds is not None and self.num_rounds > 0:
            num_rounds_to_process = min(self.num_rounds, num_rounds_to_process)

        print(f"Processing {num_rounds_to_process} rounds...")
        
        for i, round_data in enumerate(self.comparison_data[:num_rounds_to_process]):
            round_num = round_data["round_num"]
            print(f"\n[Round {round_num}/{num_rounds_to_process}] Processing...")
            
            prev_answer = round_data["answer"].strip()
            suggestions = round_data.get("suggested_questions", [])

            round_model_results = {}
            
            for name in self.model_names:
                print(f"  Evaluating model: {name}")
                source_prefix = self._get_source_prefix(name)
                selected_question = self._get_selected_question_for_source(suggestions, source_prefix)
                
                if not selected_question:
                    print(f"    No selected question found for {name}")
                    round_model_results[name] = ModelRoundResult([])
                    continue

                print(f"    Found selected question for {name}: {selected_question[:60]}{'...' if len(selected_question) > 60 else ''}")
                
                # --- Get Absolute Scores for selected question ---
                print(f"    Evaluating selected question for {name}...")
                
                metrics = await self._get_absolute_scores(selected_question, prev_answer, all_time_suggestions[name], conversation_history)
                all_time_suggestions[name].append(selected_question) # Add to history *after* scoring for originality
                
                print(f"    Completed {name}")
                
                round_model_results[name] = ModelRoundResult(
                    all_question_metrics=[metrics]  # Only the selected question
                )

            # --- Update history for next round ---
            actual_question = self.transcript[i*2]["text"] if i*2 < len(self.transcript) else ""
            conversation_history.append({"question": actual_question, "answer": prev_answer})
            
            self.evaluation_results.append(EvaluationRound(
                round_num=round_num,
                actual_question=actual_question,
                previous_answer=prev_answer,
                full_conversation_history=conversation_history.copy(),
                model_results=round_model_results,
                comparison_results={},
                comparison_feedback={}
            ))
            
            print(f"[Round {round_num}] Complete!")
        
        print("\n=== Evaluation Complete ===")

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Calculates and returns a comprehensive summary of the evaluation."""
        if not self.evaluation_results: return {}
        
        # --- Calculate Scores ---
        summary_scores = {}
        for name in self.model_names:
            all_rounds_for_model = [r.model_results.get(name) for r in self.evaluation_results if r.model_results.get(name)]
            
            all_question_metrics = [q_metric for res in all_rounds_for_model if res for q_metric in res.all_question_metrics]
            
            avg_absolute_metrics = defaultdict(float)
            if all_question_metrics:
                per_question_metric_names = [k for k in ALL_METRIC_RUBRICS.keys()]
                for metric_name in per_question_metric_names:
                    avg_absolute_metrics[f"avg_{metric_name}"] = sum(m.scores.get(metric_name, 0.0) for m in all_question_metrics) / len(all_question_metrics)

            summary_scores[name] = {
                "successful_rounds": len(all_rounds_for_model),
                "total_questions_evaluated": len(all_question_metrics),
                **avg_absolute_metrics
            }

        # --- Final Analysis ---
        analysis = {}
        
        # Calculate overall average of the core absolute scores
        avg_abs_score_components = [key for key in summary_scores.get(self.model_names[0], {}).keys() if key.startswith('avg_')]
        analysis['overall_avg_absolute_score'] = {
            name: sum(scores[comp] for comp in avg_abs_score_components) / len(avg_abs_score_components) if avg_abs_score_components else 0
            for name, scores in summary_scores.items()
        }

        # Determine winner based on category wins
        category_winners = {}
        if any(s['successful_rounds'] > 0 for s in summary_scores.values()):
            category_winners = {
                'absolute_score': max(analysis['overall_avg_absolute_score'], key=analysis['overall_avg_absolute_score'].get),
                'reliability': max(summary_scores, key=lambda n: summary_scores[n]['successful_rounds']),
            }
        
        scores = defaultdict(int)
        for cat, winner in category_winners.items():
            if winner:
                scores[winner] += 1
        
        if scores:
            overall_winner = max(scores, key=scores.get)
            winner_cats = [cat for cat, winner in category_winners.items() if winner == overall_winner]
            analysis['conclusion'] = f"{overall_winner.upper()} is the overall winner, scoring highest in {len(winner_cats)}/{len(category_winners)} categories: {winner_cats}."
        else:
            analysis['conclusion'] = "Could not determine an overall winner."

        return {
            "total_rounds_evaluated": len(self.evaluation_results),
            "final_analysis": analysis,
            "summary_scores_by_model": summary_scores,
        }

    def save_results(self, output_path: str):
        """Saves the evaluation results to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if dataclasses.is_dataclass(o):
                    return asdict(o)
                return super().default(o)
        
        results = {
            "evaluation_summary": self.get_evaluation_summary(),
            "detailed_metrics_by_round": self.evaluation_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=4)
        print(f"\nPrometheus evaluation results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Run Prometheus evaluation on enhanced comparison data (selected questions only).')
    parser.add_argument('--input-path', type=str, required=True, help='Path to the input comparison data JSON file.')
    parser.add_argument('--transcript-path', type=str, required=True, help='Path to the original transcript for context.')
    parser.add_argument('--outline-path', type=str, required=True, help='Path to the interview outline JSON file.')
    parser.add_argument('--persona-path', type=str, required=True, help='Path to the interviewer persona JSON file.')
    parser.add_argument('--output-path', type=str, help='Path to save the evaluation results JSON file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--num-rounds', type=int, default=None, help='Number of rounds to process for a quick test run.')
    parser.add_argument('--judge-model', type=str, default=None, help='Path to Prometheus model or HuggingFace model ID. Overrides PROMETHEUS_MODEL_PATH env var.')
    
    args = parser.parse_args()

    if not args.output_path:
        base_name = os.path.basename(args.input_path).replace('comparison_data_enhanced_', '')
        args.output_path = f"data/evaluations/evaluation_results_enhanced_prometheus_selected_{base_name}"

    for path in [args.input_path, args.transcript_path, args.outline_path, args.persona_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'")
            return
    
    evaluator = PrometheusEvaluatorSelectedOnly(
        comparison_data_path=args.input_path,
        transcript_path=args.transcript_path,
        outline_path=args.outline_path,
        persona_path=args.persona_path,
        verbose=args.verbose,
        num_rounds=args.num_rounds,
        judge_model=args.judge_model
    )
    await evaluator.evaluate()
    evaluator.save_results(args.output_path)
    
    summary = evaluator.get_evaluation_summary()
    print("\n--- Prometheus Evaluation Summary (Selected Questions Only) ---")
    print(json.dumps(summary, indent=2))
    print("------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())


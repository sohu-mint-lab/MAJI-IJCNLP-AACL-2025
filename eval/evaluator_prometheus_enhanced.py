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


# --- Prometheus Imports ---
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, RELATIVE_PROMPT

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
JUDGE_MODEL = "prometheus-eval/prometheus-7b-v2.0"
# --- MODELS TO BE EVALUATED ---
# This list defines which models/sources from the data file will be compared.
MODELS_TO_EVALUATE = [
    'v1', 
    'v2', 
    'v3', 
    'llm_base', 
    'llm_cot', 
    'llm_tot', 
    'llm_rag'
]

# --- RUBRICS ---
# Metric mapping with MAJI evaluator:
# - context_relevance: Same concept (how well question follows from previous answer)
# - originality: Same concept (novelty compared to previous questions)  
# - insight: Similar to MAJI's insight_category but scored instead of categorical
# - coherence: Same as MAJI's coherence (renamed from flow)
# - qualitative_elaboration: Same as MAJI's qualitative_elaboration (renamed from elaboration)
# - outline_relevance: Similar to MAJI's outline_relevance (renamed from strategic_progression)
# - persona_alignment: Same concept (tailored to interviewee's background)
# - conversational_synthesis: MAJI doesn't measure this (connecting to earlier conversation parts)

def create_score_rubric(criteria: str, descriptions: Dict[int, str]) -> str:
    """Formats the rubric data into the SCORE_RUBRIC_TEMPLATE."""
    return SCORE_RUBRIC_TEMPLATE.format(
        criteria=criteria,
        score1_description=descriptions[1],
        score2_description=descriptions[2],
        score3_description=descriptions[3],
        score4_description=descriptions[4],
        score5_description=descriptions[5]
    )

ALL_METRIC_RUBRICS = {
    "context_relevance": create_score_rubric(
        criteria="How well does the question logically follow from the interviewee's previous answer?",
        descriptions={1: "Not at all relevant.", 2: "Slightly relevant.", 3: "Moderately relevant.", 4: "Relevant.", 5: "Highly relevant."}
    ),
    "originality": create_score_rubric(
        criteria="How novel is this question compared to the history of all previously suggested questions for this specific model?",
        descriptions={1: "Identical/rephrased.", 2: "Very similar.", 3: "New angle.", 4: "Mostly new.", 5: "Completely new."}
    ),
    "insight": create_score_rubric(
        criteria="Does the question probe deeper, encouraging novel reflection?",
        descriptions={1: "Surface-level.", 2: "Asks for basic elaboration.", 3: "Encourages some reflection.", 4: "Prompts connection of ideas.", 5: "Deeply insightful."}
    ),
    "coherence": create_score_rubric(
        criteria="Does the question feel like a natural, smooth continuation of the dialogue?",
        descriptions={1: "Abrupt and jarring.", 2: "A bit awkward.", 3: "Acceptable.", 4: "Natural transition.", 5: "Very smooth."}
    ),
    "qualitative_elaboration": create_score_rubric(
        criteria="Does the question encourage the interviewee to provide a detailed, in-depth answer?",
        descriptions={1: "Elicits yes/no.", 2: "Elicits a short, factual answer.", 3: "Encourages a few sentences.", 4: "Prompts for a story/example.", 5: "Elicits a long, comprehensive answer."}
    ),
    "outline_relevance": create_score_rubric(
        criteria="Does the question creatively bridge the current dialogue with the intended interview structure (outline), or does it just bluntly repeat an outline point?",
        descriptions={
            1: "Completely ignores or contradicts the outline's direction.",
            2: "Bluntly asks a question from the outline without connecting it to the conversation.",
            3: "Loosely connects to an outline topic but the transition is awkward.",
            4: "Smoothly transitions to an outline topic, clearly building on the last answer.",
            5: "Artfully weaves an outline topic into the conversation, making the transition feel both natural and strategic."
        }
    ),
    "persona_alignment": create_score_rubric(
        criteria="How well-suited is this question to the interviewee's specific background, expertise, and known interests as described in their persona? A good question is tailored to elicit a unique and insightful answer based on the interviewee's specific experiences.",
        descriptions={
            1: "Generic question, irrelevant to the interviewee's specific persona.",
            2: "Vaguely related to the interviewee's field, but not tailored to their specific role or accomplishments.",
            3: "Asks about a topic relevant to the interviewee, but it's a standard question that doesn't probe their unique expertise.",
            4: "The question is well-tailored, touching on specific aspects of the interviewee's known experience or expertise.",
            5: "Excellent question that targets the core of the interviewee's unique expertise or perspective, making it highly likely to elicit a novel and insightful response."
        }
    ),
    "conversational_synthesis": create_score_rubric(
        criteria="Does the question connect the interviewee's most recent answer with earlier parts of the conversation, weaving together themes, or does it treat each turn as an isolated event?",
        descriptions={
            1: "Feels completely disconnected from the rest of the conversation history.",
            2: "Vaguely references something said earlier, but the connection is weak.",
            3: "Makes a simple, direct link to an immediately preceding turn.",
            4: "Connects the current answer to a broader theme discussed earlier in the conversation.",
            5: "Masterfully synthesizes multiple points from the conversation history to create a deeply contextualized and insightful question."
        }
    )
}
RELATIVE_COMPARISON_RUBRIC = "Which of the two proposed questions is a better, more insightful, and more natural follow-up to the conversation?"

# --- DATA CLASSES ---
@dataclass
class QuestionMetrics:
    question: str
    scores: Dict[str, float]
    feedback: Dict[str, str]

@dataclass
class ModelRoundResult:
    all_question_metrics: List[QuestionMetrics]

@dataclass
class EvaluationRound:
    round_num: int
    actual_question: str
    previous_answer: str
    full_conversation_history: List[Dict[str, str]]
    model_results: Dict[str, ModelRoundResult]
    comparison_results: Dict[str, str]
    comparison_feedback: Dict[str, str]

# --- MAIN EVALUATOR CLASS ---
class PrometheusEvaluator:
    def __init__(self, comparison_data_path: str, transcript_path: str, outline_path: str, persona_path: str, verbose: bool = True, num_rounds: Optional[int] = None):
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

        print(f"--- Initializing Prometheus Judge ({JUDGE_MODEL}) ---")
        self.model = VLLM(model=JUDGE_MODEL)
        self.judge = PrometheusEval(model=self.model, absolute_grade_template=ABSOLUTE_PROMPT, relative_grade_template=RELATIVE_PROMPT)
        self.evaluation_results: List[EvaluationRound] = []

    def _get_source_prefix(self, model_name: str) -> str:
        """Gets the expected source prefix string for a given model name."""
        return f'dcagent_{model_name}' if model_name in ['v1', 'v2', 'v3'] else model_name

    def _get_all_questions_for_source(self, suggestions: List[Dict], source_prefix: str) -> List[str]:
        """Gets all suggested questions for a given source."""
        return [s['question'] for s in suggestions if s.get('source', '').startswith(source_prefix) and 'question' in s]

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

    async def _get_relative_comparison(self, q_a: str, q_b: str, instruction: str, label_a: str, label_b: str) -> Tuple[str, str]:
        """Compares two questions and returns the winner and feedback."""
        try:
            feedback, winner = await asyncio.to_thread(
                self.judge.single_relative_grade,
                instruction=instruction, response_A=q_a, response_B=q_b, rubric=RELATIVE_COMPARISON_RUBRIC, reference_answer=""
            )
            return (label_a if winner == "A" else label_b if winner == "B" else "tie", feedback)
        except Exception as e:
            if self.verbose: print(f"  -> Relative comparison failed: {e}")
            return "tie", f"Error: {e}"

    async def evaluate(self):
        """Processes the entire comparison file round by round."""
        file_name = os.path.basename(self.comparison_data_path)
        print(f"\n=== Starting Enhanced Prometheus Evaluation for {file_name} ===")
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
                all_questions = self._get_all_questions_for_source(suggestions, source_prefix)
                
                if not all_questions:
                    print(f"    No questions found for {name}")
                    round_model_results[name] = ModelRoundResult([])
                    continue

                print(f"    Found {len(all_questions)} questions for {name}")
                
                # --- Get Absolute Scores for each question & calculate Fluency ---
                all_metrics_for_model = []
                
                print(f"    Evaluating individual questions for {name}...")
                for q_idx, question in enumerate(all_questions):
                    print(f"      Question {q_idx+1}/{len(all_questions)}: {question[:60]}{'...' if len(question) > 60 else ''}")
                    
                    metrics = await self._get_absolute_scores(question, prev_answer, all_time_suggestions[name], conversation_history)
                    all_metrics_for_model.append(metrics)
                    all_time_suggestions[name].append(question) # Add to history *after* scoring for originality
                
                print(f"    Completed {name}")
                
                round_model_results[name] = ModelRoundResult(
                    all_question_metrics=all_metrics_for_model
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
    parser = argparse.ArgumentParser(description='Run Prometheus evaluation on enhanced comparison data.')
    parser.add_argument('--input-path', type=str, required=True, help='Path to the input comparison data JSON file.')
    parser.add_argument('--transcript-path', type=str, required=True, help='Path to the original transcript for context.')
    parser.add_argument('--outline-path', type=str, required=True, help='Path to the interview outline JSON file.')
    parser.add_argument('--persona-path', type=str, required=True, help='Path to the interviewer persona JSON file.')
    parser.add_argument('--output-path', type=str, help='Path to save the evaluation results JSON file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--num-rounds', type=int, default=None, help='Number of rounds to process for a quick test run.')
    
    args = parser.parse_args()

    if not args.output_path:
        base_name = os.path.basename(args.input_path).replace('comparison_data_enhanced_', '')
        args.output_path = f"data/evaluations/evaluation_results_enhanced_prometheus_{base_name}"

    for path in [args.input_path, args.transcript_path, args.outline_path, args.persona_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'")
            return
    
    evaluator = PrometheusEvaluator(
        comparison_data_path=args.input_path,
        transcript_path=args.transcript_path,
        outline_path=args.outline_path,
        persona_path=args.persona_path,
        verbose=args.verbose,
        num_rounds=args.num_rounds
    )
    await evaluator.evaluate()
    evaluator.save_results(args.output_path)
    
    summary = evaluator.get_evaluation_summary()
    print("\n--- Enhanced Prometheus Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    print("------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main()) 
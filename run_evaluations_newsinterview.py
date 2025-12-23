
import os
import subprocess
import sys
import argparse
import json
from typing import List
import multiprocessing as mp
from functools import partial
import time

def find_interview_ids(transcript_dir: str) -> List[str]:
    """Finds all unique interview IDs from the transcript filenames."""
    ids = set()
    for filename in os.listdir(transcript_dir):
        if filename.endswith("_transcript.json"):
            base_id = filename.replace("_transcript.json", "")
            ids.add(base_id)
    return sorted(list(ids))

def run_command(command: List[str], interview_id: str = "unknown"):
    """Runs a command with isolated output for parallel processing."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{interview_id}] Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        print(f"[{timestamp}] [{interview_id}] Command finished successfully.")
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                print(f"[{timestamp}] [{interview_id}] Output preview:")
                for line in lines[:3]:
                    print(f"[{timestamp}] [{interview_id}]   {line}")
                print(f"[{timestamp}] [{interview_id}]   ... ({len(lines)-6} lines omitted) ...")
                for line in lines[-3:]:
                    print(f"[{timestamp}] [{interview_id}]   {line}")
            else:
                print(f"[{timestamp}] [{interview_id}] Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{timestamp}] [{interview_id}] Error running command.")
        print(f"[{timestamp}] [{interview_id}] Return Code: {e.returncode}")
        if e.stdout:
            print(f"[{timestamp}] [{interview_id}] Stdout: {e.stdout}")
        if e.stderr:
            print(f"[{timestamp}] [{interview_id}] Stderr: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[{timestamp}] [{interview_id}] Command timed out after 1 hour.")
        return False
    except FileNotFoundError:
        print(f"[{timestamp}] [{interview_id}] Error: Command not found. Ensure '{command[0]}' is in the correct path.")
        return False

def process_single_interview(interview_id: str, base_dir: str = "data", eval_types: List[str] = None):
    """Process a single interview - runs all specified evaluation types."""
    if eval_types is None:
        eval_types = ["all_gpt4o", "selected_gpt4o", "selected_prometheus"]
    
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{interview_id}] Starting evaluation...")
    
    transcript_dir = os.path.join(base_dir, "transcripts", "newsinterview")
    outline_dir = os.path.join(base_dir, "outlines", "newsinterview")
    persona_dir = os.path.join(base_dir, "personas", "newsinterview")
    evaluation_dir = os.path.join(base_dir, "evaluations")
    results_dir = os.path.join(evaluation_dir, "newsinterview_results")
    
    os.makedirs(results_dir, exist_ok=True)

    transcript_path = os.path.join(transcript_dir, f"{interview_id}_transcript.json")
    outline_path = os.path.join(outline_dir, f"{interview_id}_outline.json")
    persona_path = os.path.join(persona_dir, f"{interview_id}_persona.json")
    
    transcript_basename = os.path.basename(transcript_path)
    comparison_data_path = os.path.join(evaluation_dir, f"comparison_data_enhanced_{transcript_basename}")

    if not all(os.path.exists(p) for p in [transcript_path, outline_path, persona_path]):
        print(f"[{timestamp}] [{interview_id}] Skipping because one or more source files are missing.")
        return False
    
    # Check if comparison data exists, generate it if missing
    if not os.path.exists(comparison_data_path):
        print(f"[{timestamp}] [{interview_id}] Comparison data not found. Generating questions first...")
        generator_command = [
            sys.executable, "maji_comparison_enhanced_generator.py",
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
            "--only-v1",  # Only generate V1 questions
        ]
        if not run_command(generator_command, interview_id):
            print(f"[{timestamp}] [{interview_id}] Generator failed. Skipping evaluation.")
            return False
        print(f"[{timestamp}] [{interview_id}] Comparison data generated successfully.")
    
    results = {}
    
    # 1. GPT-4o judging all questions
    if "all_gpt4o" in eval_types:
        print(f"[{timestamp}] [{interview_id}] Running GPT-4o evaluation (all questions)...")
        evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_{transcript_basename}")
        evaluator_command = [
            sys.executable, "eval/evaluator_maji_enhanced_baseline.py",
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
        ]
        if run_command(evaluator_command, interview_id):
            final_destination = os.path.join(results_dir, f"{interview_id}_evaluation_all_gpt4o.json")
            if os.path.exists(evaluator_output_path):
                os.rename(evaluator_output_path, final_destination)
                print(f"[{timestamp}] [{interview_id}] Saved all GPT-4o results to: {final_destination}")
                results["all_gpt4o"] = True
            else:
                print(f"[{timestamp}] [{interview_id}] Could not find evaluator output.")
                results["all_gpt4o"] = False
        else:
            results["all_gpt4o"] = False
    
    # 2. GPT-4o judging selected questions
    if "selected_gpt4o" in eval_types:
        print(f"[{timestamp}] [{interview_id}] Running GPT-4o evaluation (selected questions only)...")
        evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_selected_{transcript_basename}")
        evaluator_command = [
            sys.executable, "eval/evaluator_maji_selected_only.py",
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
        ]
        if run_command(evaluator_command, interview_id):
            final_destination = os.path.join(results_dir, f"{interview_id}_evaluation_selected_gpt4o.json")
            if os.path.exists(evaluator_output_path):
                os.rename(evaluator_output_path, final_destination)
                print(f"[{timestamp}] [{interview_id}] Saved selected GPT-4o results to: {final_destination}")
                results["selected_gpt4o"] = True
            else:
                print(f"[{timestamp}] [{interview_id}] Could not find evaluator output.")
                results["selected_gpt4o"] = False
        else:
            results["selected_gpt4o"] = False
    
    # 3. Prometheus 2 judging selected questions
    if "selected_prometheus" in eval_types:
        print(f"[{timestamp}] [{interview_id}] Running Prometheus 2 evaluation (selected questions only)...")
        evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_prometheus_selected_{transcript_basename}")
        evaluator_command = [
            sys.executable, "eval/evaluator_prometheus_selected_only.py",
            "--input-path", comparison_data_path,
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
            "--output-path", evaluator_output_path,
        ]
        # Add judge-model argument if environment variable is set
        judge_model = os.getenv("PROMETHEUS_MODEL_PATH")
        if judge_model:
            evaluator_command.extend(["--judge-model", judge_model])
        if run_command(evaluator_command, interview_id):
            final_destination = os.path.join(results_dir, f"{interview_id}_evaluation_selected_prometheus.json")
            if os.path.exists(evaluator_output_path):
                os.rename(evaluator_output_path, final_destination)
                print(f"[{timestamp}] [{interview_id}] Saved Prometheus results to: {final_destination}")
                results["selected_prometheus"] = True
            else:
                print(f"[{timestamp}] [{interview_id}] Could not find evaluator output.")
                results["selected_prometheus"] = False
        else:
            results["selected_prometheus"] = False

    # 4. Prometheus 2 judging all questions
    if "all_prometheus" in eval_types:
        print(f"[{timestamp}] [{interview_id}] Running Prometheus 2 evaluation (all questions)...")
        
        base_name = os.path.basename(comparison_data_path).replace('comparison_data_enhanced_', '')
        evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_prometheus_{base_name}")

        evaluator_command = [
            sys.executable, "eval/evaluator_prometheus_enhanced.py",
            "--input-path", comparison_data_path,
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
            "--output-path", evaluator_output_path,
        ]
        
        if run_command(evaluator_command, interview_id):
            final_destination = os.path.join(results_dir, f"{interview_id}_evaluation_all_prometheus.json")
            if os.path.exists(evaluator_output_path):
                os.rename(evaluator_output_path, final_destination)
                print(f"[{timestamp}] [{interview_id}] Saved all Prometheus results to: {final_destination}")
                results["all_prometheus"] = True
            else:
                print(f"[{timestamp}] [{interview_id}] Could not find evaluator output for all_prometheus at {evaluator_output_path}.")
                results["all_prometheus"] = False
        else:
            results["all_prometheus"] = False
            
    return all(results.values()) if results else False

def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluations on newsinterview test data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--limit', type=int, default=None, help="Optional: Limit the number of interviews to process."
    )
    parser.add_argument(
        '--workers', type=int, default=2, help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        '--eval-types', nargs='+', 
        choices=['all_gpt4o', 'selected_gpt4o', 'selected_prometheus', 'all_prometheus'],
        default=['all_gpt4o', 'selected_gpt4o'],
        help="Which evaluation types to run (default: all_gpt4o, selected_gpt4o - excluding prometheus)"
    )
    args = parser.parse_args()

    base_dir = "data"
    transcript_dir = os.path.join(base_dir, "transcripts", "newsinterview")
    evaluation_dir = os.path.join(base_dir, "evaluations")
    results_dir = os.path.join(evaluation_dir, "newsinterview_results")
    os.makedirs(results_dir, exist_ok=True)

    interview_ids = find_interview_ids(transcript_dir)
    if args.limit:
        interview_ids = interview_ids[:args.limit]
    
    print(f"--- Found {len(interview_ids)} interviews to process with {args.workers} workers. ---")
    print(f"--- Running evaluation types: {args.eval_types} ---")

    # Create a partial function with the base_dir and eval_types parameters
    process_func = partial(process_single_interview, base_dir=base_dir, eval_types=args.eval_types)
    
    # Use multiprocessing to run interviews in parallel
    start_time = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(process_func, interview_ids)
    
    end_time = time.time()
    successful = sum(results)
    total_time = end_time - start_time
    
    print(f"\n--- Full Evaluation Processing Complete ---")
    print(f"Successfully processed: {successful}/{len(interview_ids)} interviews")
    print(f"Total time: {total_time/60:.1f} minutes")
    if interview_ids:
        print(f"Average time per interview: {total_time/len(interview_ids):.1f} seconds")

if __name__ == "__main__":
    main()


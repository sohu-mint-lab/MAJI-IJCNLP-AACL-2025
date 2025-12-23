import os
import subprocess
import sys
import argparse
import json
from typing import List
import multiprocessing as mp
from functools import partial
import logging
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
        # Capture output to prevent mixing with other processes
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout per command (increased from 30 min)
        )
        print(f"[{timestamp}] [{interview_id}] Pipeline completed successfully.")
        # Show a summary of the output (first and last few lines)
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

def process_single_interview(interview_id: str, base_dir: str = "data"):
    """Process a single interview - designed to be run in parallel."""
    # Set up process-specific logging
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{interview_id}] Starting processing...")
    
    transcript_dir = os.path.join(base_dir, "transcripts", "newsinterview")
    outline_dir = os.path.join(base_dir, "outlines", "newsinterview")
    persona_dir = os.path.join(base_dir, "personas", "newsinterview")
    evaluation_dir = os.path.join(base_dir, "evaluations")
    results_dir = os.path.join(evaluation_dir, "newsinterview_results")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    transcript_path = os.path.join(transcript_dir, f"{interview_id}_transcript.json")
    outline_path = os.path.join(outline_dir, f"{interview_id}_outline.json")
    persona_path = os.path.join(persona_dir, f"{interview_id}_persona.json")
    evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_{os.path.basename(transcript_path)}")

    if not all(os.path.exists(p) for p in [transcript_path, outline_path, persona_path]):
        print(f"[{timestamp}] [{interview_id}] Skipping because one or more source files are missing.")
        return False
    
    # Run generator
    print(f"[{timestamp}] [{interview_id}] Starting generator (this may take 10-30 minutes)...")
    generator_command = [
        sys.executable, "maji_comparison_enhanced_generator.py",
        "--transcript-path", transcript_path,
        "--outline-path", outline_path,
        "--persona-path", persona_path,
    ]
    if not run_command(generator_command, interview_id):
        print(f"[{timestamp}] [{interview_id}] Generator failed. Skipping evaluation.")
        return False
    
    # Run evaluator
    print(f"[{timestamp}] [{interview_id}] Starting evaluations...")
    evaluator_command = [
        sys.executable, "eval/evaluator_maji_enhanced_baseline.py",
        "--transcript-path", transcript_path,
        "--outline-path", outline_path,
        "--persona-path", persona_path,
    ]
    if not run_command(evaluator_command, interview_id):
        print(f"[{timestamp}] [{interview_id}] Evaluator failed.")
        return False
    
    # Move result to final location
    final_destination = os.path.join(results_dir, f"{interview_id}_evaluation_result.json")
    if os.path.exists(evaluator_output_path):
        os.rename(evaluator_output_path, final_destination)
        print(f"[{timestamp}] [{interview_id}] Moved final result to: {final_destination}")
        return True
    else:
        print(f"[{timestamp}] [{interview_id}] Could not find evaluator output at {evaluator_output_path}.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run the full MAJI generation and evaluation pipeline using the enhanced generator.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--limit', type=int, default=None, help="Optional: Limit the number of interviews to process."
    )
    parser.add_argument(
        '--workers', type=int, default=2, help="Number of parallel workers (default: 2, reduced for stability)"
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
    print(f"--- Using reduced worker count for stability with LLM calls ---")

    # Create a partial function with the base_dir parameter
    process_func = partial(process_single_interview, base_dir=base_dir)
    
    # Use multiprocessing to run interviews in parallel
    start_time = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(process_func, interview_ids)
    
    end_time = time.time()
    successful = sum(results)
    total_time = end_time - start_time
    
    print(f"\n--- Full Pipeline Processing Complete ---")
    print(f"Successfully processed: {successful}/{len(interview_ids)} interviews")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per interview: {total_time/len(interview_ids):.1f} seconds")

if __name__ == "__main__":
    main() 
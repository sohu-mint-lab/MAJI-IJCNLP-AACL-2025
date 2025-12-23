import os
import subprocess
import sys
import argparse
import json
from typing import List

def find_interview_ids(transcript_dir: str) -> List[str]:
    """Finds all unique interview IDs from the transcript filenames."""
    ids = set()
    for filename in os.listdir(transcript_dir):
        if filename.endswith("_transcript.json"):
            base_id = filename.replace("_transcript.json", "")
            ids.add(base_id)
    return sorted(list(ids))

def run_command(command: List[str]):
    """Runs a command, streaming its output directly to the console."""
    print(f"\nRunning command: {' '.join(command)}")
    try:
        # Using Popen to allow for more flexible process management,
        # but waiting for it to complete to keep the streaming output behavior.
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode == 0:
            print("Command finished successfully.")
            return True
        else:
            print(f"Error running command.")
            print(f"  Return Code: {process.returncode}")
            print("  --- Check console output above for error details ---")
            return False
    except FileNotFoundError:
        print(f"Error: Command not found. Ensure '{command[0]}' is in the correct path.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
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
        '--skip-generation', action='store_true', help="Skip the comparison data generation step and use existing files."
    )
    args = parser.parse_args()

    base_dir = "data"
    transcript_dir = os.path.join(base_dir, "transcripts", "newsinterview")
    outline_dir = os.path.join(base_dir, "outlines", "newsinterview")
    persona_dir = os.path.join(base_dir, "personas", "newsinterview")
    evaluation_dir = os.path.join(base_dir, "evaluations")
    results_dir = os.path.join(evaluation_dir, "newsinterview_results")
    os.makedirs(results_dir, exist_ok=True)

    interview_ids = find_interview_ids(transcript_dir)
    if args.limit:
        interview_ids = interview_ids[:args.limit]
    print(f"--- Found {len(interview_ids)} interviews to process. ---")

    for i, interview_id in enumerate(interview_ids):
        print(f"\n{'='*20} Processing Interview {i+1}/{len(interview_ids)}: {interview_id} {'='*20}")

        transcript_path = os.path.join(transcript_dir, f"{interview_id}_transcript.json")
        transcript_filename = os.path.basename(transcript_path)
        outline_path = os.path.join(outline_dir, f"{interview_id}_outline.json")
        persona_path = os.path.join(persona_dir, f"{interview_id}_persona.json")
        
        comparison_data_path = os.path.join(evaluation_dir, f"comparison_data_enhanced_{transcript_filename}")
        maji_evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_{transcript_filename}")
        prometheus_evaluator_output_path = os.path.join(evaluation_dir, f"evaluation_results_enhanced_prometheus_{transcript_filename}")


        if not all(os.path.exists(p) for p in [transcript_path, outline_path, persona_path]):
            print(f"Skipping {interview_id} because one or more source files are missing.")
            continue
        
        # Step 1: Run the generator and wait for it to complete (optional).
        if not args.skip_generation:
            generator_command = [
                sys.executable, "maji_comparison_enhanced_generator.py",
                "--transcript-path", transcript_path,
                "--outline-path", outline_path,
                "--persona-path", persona_path,
            ]
            if not run_command(generator_command):
                print(f"Generator failed for {interview_id}. Skipping evaluation.")
                continue
        else:
            print(f"Skipping generation for {interview_id} as requested.")
            
        if not os.path.exists(comparison_data_path):
            print(f"Comparison data file '{comparison_data_path}' not found. Skipping evaluation.")
            continue
        
        # Step 2: Run the Prometheus evaluator.
        print(f"\nStarting Prometheus evaluator for {interview_id}...")

        prometheus_evaluator_command = [
            sys.executable, "eval/evaluator_prometheus_enhanced.py",
            "--input-path", comparison_data_path,
            "--transcript-path", transcript_path,
            "--outline-path", outline_path,
            "--persona-path", persona_path,
            "--output-path", prometheus_evaluator_output_path,
        ]
        
        prometheus_success = run_command(prometheus_evaluator_command)

        # Step 3: Move the results to their final destination.
        if prometheus_success:
            final_prometheus_destination = os.path.join(results_dir, f"{interview_id}_prometheus_evaluation_result.json")
            if os.path.exists(prometheus_evaluator_output_path):
                os.rename(prometheus_evaluator_output_path, final_prometheus_destination)
                print(f"Moved Prometheus result to: {final_prometheus_destination}")
            else:
                print(f"Could not find Prometheus evaluator output at {prometheus_evaluator_output_path}.")


    print("\n--- Full Pipeline Processing Complete ---")

if __name__ == "__main__":
    main() 
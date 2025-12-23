#!/usr/bin/env python3
# evaluate_prometheus_results.py

import os
import json
from collections import defaultdict
import numpy as np

try:
    from scipy.stats import ttest_rel, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("SciPy not found – will skip t-test/Wilcoxon.")

def find_and_collect_metrics(data_dict, collected):
    for key, value in data_dict.items():
        if isinstance(value, (int, float)):
            collected[key].append(float(value))
        elif isinstance(value, dict):
            find_and_collect_metrics(value, collected)

def bootstrap_ci(diff_vec, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff_vec, (n_boot, len(diff_vec)), replace=True).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])

def analyze_evaluation_results(directory, baseline_model="llm_base"):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return

    model_data = defaultdict(lambda: defaultdict(list))  # model → metric → list of values
    files_processed = 0

    for filename in os.listdir(directory):
        # Modified to match the actual file naming pattern
        if not filename.endswith("_selected_prometheus.json"):
            continue
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            summary_scores = data.get("evaluation_summary", {}).get("summary_scores_by_model", {})
            if not isinstance(summary_scores, dict):
                print(f"Skipping '{filename}': summary_scores_by_model missing/invalid.")
                continue

            files_processed += 1
            for model, metrics in summary_scores.items():
                find_and_collect_metrics(metrics, model_data[model])

        except Exception as e:
            print(f"Error in '{filename}': {e}")

    if not files_processed:
        print("No valid files processed.")
        return

    # Compute means, summing specific keys
    avg_results = {}
    for m, m_dict in model_data.items():
        avg_results[m] = {}
        for k, vs in m_dict.items():
            if k in ["successful_rounds", "total_questions_evaluated"]:
                avg_results[m][k] = np.sum(vs)
            else:
                avg_results[m][k] = np.mean(vs)

    # Find best
    best_of = {}
    for m, m_dict in avg_results.items():
        for k, v in m_dict.items():
            if k not in best_of:
                best_of[k] = v
            else:
                if 'redundancy' in k:
                    if v < best_of[k]:
                        best_of[k] = v
                else:
                    if v > best_of[k]:
                        best_of[k] = v

    # Print results
    print("\n--- Prometheus Evaluation Summary ---")
    for model, m_dict in avg_results.items():
        print(f"Model: {model}")
        print("-" * 60)
        for metric, mean_val in sorted(m_dict.items()):
            is_best = (mean_val == best_of[metric])
            highlight = " (best)" if is_best else ""

            stats = ""
            if model != baseline_model and baseline_model in model_data and metric in model_data[baseline_model]:
                base_vals = np.array(model_data[baseline_model][metric])
                comp_vals = np.array(model_data[model][metric])
                if len(base_vals) != len(comp_vals):
                    stats = f" (unequal n: {len(comp_vals)} vs {len(base_vals)})"
                else:
                    diff = comp_vals - base_vals
                    delta = diff.mean()
                    ci_lo, ci_hi = bootstrap_ci(diff)
                    stats = f" Δ={delta:+.3f} 95%CI[{ci_lo:.3f},{ci_hi:.3f}]"
                    if HAVE_SCIPY:
                        t_p = ttest_rel(comp_vals, base_vals).pvalue
                        w_p = wilcoxon(comp_vals, base_vals).pvalue
                        stats += f" p_t={t_p:.3g} p_w={w_p:.3g}"

            print(f"  {metric}: {mean_val:.4f}{highlight}{stats}")
        print("=" * 60)

    print(f"\nProcessed {files_processed} files. Baseline: {baseline_model}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "data/evaluations/newsinterview_results"
    analyze_evaluation_results(results_dir, baseline_model="llm_base")

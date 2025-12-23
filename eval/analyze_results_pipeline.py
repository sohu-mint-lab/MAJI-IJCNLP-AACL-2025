#!/usr/bin/env python3
# analyze_results_pipeline.py
"""
Analyse MAJI (or compatible) evaluation JSONs.

• Keeps original outputs: best-score, redundancy-is-lower, coverage, specialist
  win-rates, insight profile & trajectory.
• Adds statistical tests vs. a configurable baseline:
    – Δ̄ (mean paired difference)
    – 95 % bootstrap CI
    – paired t-test & Wilcoxon (if SciPy installed)
• Gracefully skips/flags metrics whose observation counts differ.
• NEW: Adjusts originality scores using threshold-based normalization.
"""

import argparse, json, os
from collections import defaultdict
from typing import Dict, List

import numpy as np
try:
    from scipy.stats import ttest_rel, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print('SciPy not found – t-test/Wilcoxon will be omitted.')

# ───────────────────────── helpers ──────────────────────────
def deep_collect_numeric(d: Dict, out: Dict[str, List[float]], prefix: str = ""):
    """flatten nested dict, appending numeric leaves into out[key]."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            out[key].append(float(v))
        elif isinstance(v, dict):
            deep_collect_numeric(v, out, key)

def bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5])

def adjust_originality_scores(originality_values: List[float], 
                            similarity_threshold: float = 0.6,
                            adjustment_method: str = "threshold") -> List[float]:
    """
    Adjusts originality scores to be more reasonable.
    
    The original formula was: originality = 1.0 - max_similarity
    This creates artificially low scores because:
    1. Cosine similarity between different questions is often 0.3-0.7
    2. The formula assumes high similarity = low originality, but doesn't account for baseline similarity
    
    Args:
        originality_values: List of original originality scores (0-1)
        similarity_threshold: Threshold above which questions are considered similar (default 0.6)
        adjustment_method: Method to use for adjustment ("threshold", "squared", "normalized")
    
    Returns:
        List of adjusted originality scores
    """
    adjusted = []
    
    for orig_score in originality_values:
        # Convert back to max_similarity (original formula: orig = 1.0 - max_sim)
        max_similarity = 1.0 - orig_score
        
        if adjustment_method == "threshold":
            # Threshold-based: questions below threshold are considered original
            if max_similarity < similarity_threshold:
                adjusted_score = 1.0  # Completely original
            else:
                # Scale the remaining similarity range
                adjusted_score = 1.0 - (max_similarity - similarity_threshold) / (1.0 - similarity_threshold)
                adjusted_score = max(0.0, min(1.0, adjusted_score))  # Clip to [0,1]
                
        elif adjustment_method == "squared":
            # Square the similarity to penalize high values more
            adjusted_score = 1.0 - (max_similarity ** 2)
            
        elif adjustment_method == "normalized":
            # Normalize by expected baseline similarity
            baseline_similarity = 0.4  # Expected similarity between different questions
            if max_similarity <= baseline_similarity:
                adjusted_score = 1.0
            else:
                adjusted_score = max(0.0, (baseline_similarity - max_similarity) / baseline_similarity)
                
        else:
            # No adjustment
            adjusted_score = orig_score
            
        adjusted.append(adjusted_score)
    
    return adjusted

# ───────────────────────── main ────────────────────────────
def analyze(results_dir: str, baseline: str = "llm_base", 
           adjust_originality: bool = True, 
           originality_threshold: float = 0.6,
           originality_method: str = "threshold"):

    if not os.path.isdir(results_dir):
        raise SystemExit(f"directory not found: {results_dir}")

    # model → analysis_type → metric → list[values]
    eval_vecs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # coverage
    cov_vecs  = defaultdict(lambda: defaultdict(list))
    # specialist contributions
    spec_tot  = defaultdict(lambda: defaultdict(float))

    files_processed = 0
    files_skipped = 0
    
    for fn in os.listdir(results_dir):
        # Only process LLM evaluation files, skip Prometheus files
        if not fn.endswith("_evaluation_result.json") or "prometheus" in fn:
            continue
        try:
            data = json.load(open(os.path.join(results_dir, fn)))
        except Exception as e:
            print(f"{fn}: {e}")
            files_skipped += 1
            continue

        # evaluation_summary --------------------------------------------------
        summary = data.get("evaluation_summary")
        if isinstance(summary, dict):
            try:
                files_processed += 1
                for model, analyses in summary.items():
                    if not isinstance(analyses, dict):
                        print(f"{fn}: model {model} has non-dict analyses: {type(analyses)}")
                        continue
                    for ana_type, metrics in analyses.items():
                        if not isinstance(metrics, dict):
                            print(f"{fn}: model {model}, analysis {ana_type} has non-dict metrics: {type(metrics)}")
                            continue
                        deep_collect_numeric(metrics, eval_vecs[model][ana_type])
                        # specialist win-rate (keys look like win_rate_<name>)
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and k.startswith("win_rate_"):
                                spec_tot[model][k.replace("win_rate_", "")] += v
            except Exception as e:
                print(f"{fn}: error processing evaluation_summary: {e}")
                files_skipped += 1
                continue

        # final_coverage ------------------------------------------------------
        cov = data.get("final_coverage")
        if isinstance(cov, dict):
            for model, metrics in cov.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            cov_vecs[model][k].append(float(v))

    if files_processed == 0:
        raise SystemExit("no valid JSONs parsed.")

    # Adjust originality scores if requested
    if adjust_originality:
        print(f"\nAdjusting originality scores using {originality_method} method (threshold={originality_threshold})")
        for model in eval_vecs:
            for ana_type in eval_vecs[model]:
                if "avg_originality" in eval_vecs[model][ana_type]:
                    original_scores = eval_vecs[model][ana_type]["avg_originality"]
                    adjusted_scores = adjust_originality_scores(
                        original_scores, 
                        similarity_threshold=originality_threshold,
                        adjustment_method=originality_method
                    )
                    eval_vecs[model][ana_type]["avg_originality"] = adjusted_scores
                    print(f"  {model}.{ana_type}: avg_originality adjusted from {np.mean(original_scores):.3f} to {np.mean(adjusted_scores):.3f}")

    # mean tables ------------------------------------------------------------
    mean_eval = {m: {a: {k: np.mean(v) for k, v in md.items()}
                     for a, md in ad.items()}
                 for m, ad in eval_vecs.items()}

    # best values (low better for redundancy)
    best_of = defaultdict(dict)
    for m, ad in mean_eval.items():
        for a, md in ad.items():
            for k, v in md.items():
                better = (v < best_of[a].get(k, 1e9)) if "redundancy" in k \
                         else (v > best_of[a].get(k, -1e9))
                if better:
                    best_of[a][k] = v

    # constants we skip from stats
    SKIP_CONST = {"successful_rounds", "total_questions_evaluated"}

    print("\nEvaluation Summary (best; CIs & *p* vs baseline)")
    if adjust_originality:
        print(f"Originality scores adjusted using {originality_method} method")
    print("="*100)
    for model in sorted(mean_eval):
        print(f"Model: {model}")
        print("-"*100)
        for ana, md in sorted(mean_eval[model].items()):
            print(f"  Analysis: {ana}")
            for metric, mean_val in sorted(md.items()):
                highlight = " (best)" if mean_val == best_of[ana][metric] else ""
                stats_txt = ""

                # significance
                if model != baseline and metric not in SKIP_CONST \
                   and baseline in eval_vecs and metric in eval_vecs[baseline][ana]:
                    base = np.array(eval_vecs[baseline][ana][metric])
                    comp = np.array(eval_vecs[model][ana][metric])

                    if len(base) == len(comp):
                        if np.allclose(base, comp):
                            stats_txt = " (identical)"
                        else:
                            diff = comp - base
                            delta = diff.mean()
                            lo, hi = bootstrap_ci(diff)
                            stats_txt = f" Δ={delta:+.3f} CI[{lo:.3f},{hi:.3f}]"
                            if HAVE_SCIPY:
                                stats_txt += (f" p_t={ttest_rel(comp, base).pvalue:.3g}"
                                              f" p_w={wilcoxon(comp, base).pvalue:.3g}")
                    else:
                        stats_txt = f" (n mismatch {len(comp)} vs {len(base)})"

                print(f"    {metric}: {mean_val:.4f}{highlight}{stats_txt}")
            print("  " + "-"*95)
        print("="*100)

    # -------------------- coverage -----------------------------------------
    if cov_vecs:
        mean_cov = {m: {k: np.mean(v) for k, v in md.items()}
                    for m, md in cov_vecs.items()}
        best_cov  = max((d.get("coverage_percentage", 0) for d in mean_cov.values()), default=0)
        best_wc   = max((d.get("weighted_coverage_score", 0) for d in mean_cov.values()), default=0)

        print("\nCoverage Summary")
        print("="*100)
        for m, d in sorted(mean_cov.items(),
                           key=lambda x: x[1].get("weighted_coverage_score", 0), reverse=True):
            cov_hi  = " (best)" if d.get("coverage_percentage") == best_cov else ""
            wcov_hi = " (best)" if d.get("weighted_coverage_score") == best_wc else ""
            print(f"{m}: cov={d.get('coverage_percentage',0):.1f}%{cov_hi} "
                  f"w_cov={d.get('weighted_coverage_score',0):.1f}%{wcov_hi} "
                  f"(covered={d.get('covered_questions',0):.0f}/"
                  f"{d.get('total_questions',0):.0f})")
        print("="*100)

    # -------------------- specialist contributions --------------------------
    if any(spec_tot.values()):
        print("\nSpecialist Contribution Rates")
        print("="*100)
        for m, specs in sorted(spec_tot.items()):
            tot = sum(specs.values()) or 1
            print(f"{m}:")
            for spec, val in sorted(specs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {spec}: {val/tot:.2%}")
            print("-"*100)

    print(f"\nprocessed {files_processed} files  |  skipped {files_skipped} files  |  baseline = {baseline}")

# ───────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="folder with *.json evaluation files")
    ap.add_argument("--baseline", default="llm_base",
                    help="model to serve as control (default llm_base)")
    ap.add_argument("--no-adjust-originality", action="store_true",
                    help="disable originality score adjustment")
    ap.add_argument("--originality-threshold", type=float, default=0.6,
                    help="similarity threshold for originality adjustment (default 0.6)")
    ap.add_argument("--originality-method", choices=["threshold", "squared", "normalized"], 
                    default="threshold",
                    help="method for adjusting originality scores (default threshold)")
    args = ap.parse_args()
    analyze(args.results_dir, baseline=args.baseline, 
           adjust_originality=not args.no_adjust_originality,
           originality_threshold=args.originality_threshold,
           originality_method=args.originality_method)

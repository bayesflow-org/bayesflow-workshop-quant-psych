"""
Check BayesFlow diagnostics against house thresholds.

Reads a saved diagnostics DataFrame (CSV) produced by
``workflow.compute_default_diagnostics(test_data=..., as_data_frame=True)``
and produces a structured pass/fail report.

Save the DataFrame before running this script:

    metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=True)
    metrics.to_csv("metrics.csv")

Usage:
    python check_diagnostics.py --metrics metrics.csv
    python check_diagnostics.py --metrics metrics.csv --output report.json
"""

import argparse
import json
import sys

import pandas as pd


# ── House thresholds ──────────────────────────────────────────
# These match the "House thresholds" section of the skill.
ECE_EXCELLENT = 0.05
ECE_ACCEPTABLE = 0.10

NRMSE_GOOD = 0.05
NRMSE_ACCEPTABLE = 0.10
NRMSE_WEAK = 0.15

CONTRACTION_LOW = 0.80
CONTRACTION_HIGH = 0.99


def check_diagnostics(metrics: pd.DataFrame) -> dict:
    """Check a diagnostics DataFrame against house thresholds.

    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame as returned by
        ``workflow.compute_default_diagnostics(as_data_frame=True)``.
        Rows are metric names (``RMSE``, ``NRMSE``, ``Log-gamma``,
        ``ECE``, ``Post. Contraction``), columns are parameter names.

    Returns
    -------
    dict
        Structured report with per-parameter verdicts and an overall
        go / no-go decision.
    """
    report: dict = {"parameters": {}, "overall": {}}
    any_fail = False
    any_warn = False

    # Determine which RMSE row is available
    has_nrmse = "NRMSE" in metrics.index
    has_rmse = "RMSE" in metrics.index
    rmse_key = "NRMSE" if has_nrmse else ("RMSE" if has_rmse else None)

    for param in metrics.columns:
        param_report: dict = {"verdicts": []}

        # ── ECE ───────────────────────────────────────────────
        if "ECE" in metrics.index:
            ece = float(metrics.loc["ECE", param])
            param_report["ECE"] = ece
            if ece > ECE_ACCEPTABLE:
                param_report["verdicts"].append(f"FAIL: ECE = {ece:.3f} > {ECE_ACCEPTABLE}")
                any_fail = True
            elif ece > ECE_EXCELLENT:
                param_report["verdicts"].append(f"WARN: ECE = {ece:.3f} — acceptable but not excellent")
                any_warn = True
            else:
                param_report["verdicts"].append(f"OK: ECE = {ece:.3f}")

        # ── NRMSE / RMSE ─────────────────────────────────────
        if rmse_key is not None:
            nrmse = float(metrics.loc[rmse_key, param])
            param_report[rmse_key] = nrmse
            if nrmse > NRMSE_WEAK:
                param_report["verdicts"].append(f"FAIL: {rmse_key} = {nrmse:.3f} > {NRMSE_WEAK}")
                any_fail = True
            elif nrmse > NRMSE_ACCEPTABLE:
                param_report["verdicts"].append(f"WARN: {rmse_key} = {nrmse:.3f} — weak recovery")
                any_warn = True
            elif nrmse > NRMSE_GOOD:
                param_report["verdicts"].append(f"OK: {rmse_key} = {nrmse:.3f} — acceptable")
            else:
                param_report["verdicts"].append(f"OK: {rmse_key} = {nrmse:.3f}")

        # ── Posterior contraction ─────────────────────────────
        if "Post. Contraction" in metrics.index:
            contraction = float(metrics.loc["Post. Contraction", param])
            param_report["Post. Contraction"] = contraction
            ece_val = param_report.get("ECE", 0.0)

            if contraction > CONTRACTION_HIGH and ece_val > ECE_EXCELLENT:
                param_report["verdicts"].append(
                    f"FAIL: Contraction = {contraction:.3f} with ECE = {ece_val:.3f} "
                    "— overconfident and miscalibrated"
                )
                any_fail = True
            elif contraction > CONTRACTION_HIGH:
                param_report["verdicts"].append(
                    f"OK: Contraction = {contraction:.3f} — very high but calibration is good"
                )
            elif contraction < CONTRACTION_LOW:
                param_report["verdicts"].append(
                    f"WARN: Contraction = {contraction:.3f} — weak information gain"
                )
                any_warn = True
            else:
                param_report["verdicts"].append(f"OK: Contraction = {contraction:.3f}")

        report["parameters"][param] = param_report

    # ── Overall ───────────────────────────────────────────────
    if any_fail:
        decision = "STOP"
        recommendation = (
            "One or more parameters exceed hard thresholds. "
            "Do not proceed to real-data inference. "
            "Diagnose and fix first (see 'When things go wrong' table in the skill)."
        )
    elif any_warn:
        decision = "WARN"
        recommendation = (
            "Some parameters have marginal diagnostics. "
            "Proceed only if the user accepts the risk."
        )
    else:
        decision = "GO"
        recommendation = "All diagnostics within acceptable bounds. Safe to proceed."

    report["overall"] = {
        "decision": decision,
        "any_fail": any_fail,
        "any_warn": any_warn,
        "recommendation": recommendation,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Check BayesFlow diagnostics against house thresholds")
    parser.add_argument(
        "--metrics", required=True, help="Path to diagnostics CSV (saved from metrics.to_csv(...))"
    )
    parser.add_argument(
        "--output", default=None, help="Path to save JSON report (default: print to stdout)"
    )
    args = parser.parse_args()

    try:
        metrics = pd.read_csv(args.metrics, index_col=0)
    except Exception as e:
        print(json.dumps({"error": f"Could not load metrics CSV: {e}"}))
        sys.exit(1)

    # Print the table so it's visible in terminal output
    print("=== Diagnostics Table ===")
    print(metrics.to_string())
    print("=========================\n")

    report = check_diagnostics(metrics)

    output = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()

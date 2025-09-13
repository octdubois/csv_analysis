"""Command line interface for CSV Insight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bench import run_benchmark
from .io_utils import load_json, save_json
from .plan_runner import run_plan
from .report_runner import generate_insight_report
from .logging_conf import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="csvi")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--plan", required=True)
    run_p.add_argument("--data", required=True)
    run_p.add_argument("--out", required=True)

    report_p = sub.add_parser("report")
    report_p.add_argument("--analysis", required=True)
    report_p.add_argument("--out", required=True)

    bench_p = sub.add_parser("bench")
    bench_p.add_argument("--rows", type=int, default=1_000_000)
    bench_p.add_argument("--out", required=True)

    args = parser.parse_args()
    logger = configure_logging()

    if args.cmd == "run":
        plan = load_json(args.plan)
        results = run_plan(plan)
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        save_json(outdir / "analysis.json", {k: v for k, v in results.items() if k != "data"})
        if results.get("data") is not None:
            results["data"].to_csv(outdir / "data_processed.csv", index=False)
    elif args.cmd == "report":
        analysis = load_json(args.analysis)
        report = generate_insight_report(analysis)
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        save_json(outdir / "report.json", report)
        (outdir / "report.md").write_text(report["markdown"])
    elif args.cmd == "bench":
        res = run_benchmark(args.rows)
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        save_json(outdir / "bench.json", res)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cases_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in report.get("cases", [])}


def get_assertion_pass_rate(
    cases: dict[str, dict[str, Any]], evaluator: str,
) -> float | None:
    values: list[bool] = []
    for case in cases.values():
        assertion = case.get("assertions", {}).get(evaluator)
        if assertion is not None:
            values.append(assertion["value"])
    if not values:
        return None
    return sum(values) / len(values)


def get_score_average(
    cases: dict[str, dict[str, Any]], scorer: str,
) -> float | None:
    values: list[float] = []
    for case in cases.values():
        score = case.get("scores", {}).get(scorer)
        if score is not None:
            values.append(score["value"])
    if not values:
        return None
    return sum(values) / len(values)


def all_evaluator_names(
    cases_a: dict[str, dict[str, Any]], cases_b: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    assertions: set[str] = set()
    scores: set[str] = set()
    for cases in (cases_a, cases_b):
        for case in cases.values():
            assertions.update(case.get("assertions", {}).keys())
            scores.update(case.get("scores", {}).keys())
    return sorted(assertions), sorted(scores)


def print_header(
    meta_a: dict[str, Any] | None,
    meta_b: dict[str, Any] | None,
    path_a: Path,
    path_b: Path,
) -> None:
    print(f"{'Baseline:':<14} {path_a.name}")
    print(f"{'Comparison:':<14} {path_b.name}")
    print()

    if not meta_a and not meta_b:
        return

    meta_a = meta_a or {}
    meta_b = meta_b or {}
    all_keys = sorted(set(meta_a.keys()) | set(meta_b.keys()))
    for key in all_keys:
        va = meta_a.get(key, "—")
        vb = meta_b.get(key, "—")
        changed = " *" if va != vb else ""
        print(f"  {key:<12} {va} -> {vb}{changed}")
    print()


def print_summary(
    cases_a: dict[str, dict[str, Any]],
    cases_b: dict[str, dict[str, Any]],
) -> None:
    print("Summary")
    print("-" * 50)

    assertion_names, score_names = all_evaluator_names(cases_a, cases_b)

    for name in assertion_names:
        rate_a = get_assertion_pass_rate(cases_a, name)
        rate_b = get_assertion_pass_rate(cases_b, name)
        if rate_a is None and rate_b is None:
            continue
        sa = f"{rate_a:.1%}" if rate_a is not None else "—"
        sb = f"{rate_b:.1%}" if rate_b is not None else "—"
        delta = ""
        if rate_a is not None and rate_b is not None:
            d = rate_b - rate_a
            sign = "+" if d >= 0 else ""
            delta = f"  ({sign}{d:.1%})"
        print(f"  {name:<24} {sa:>7} -> {sb:<7}{delta}")

    for name in score_names:
        avg_a = get_score_average(cases_a, name)
        avg_b = get_score_average(cases_b, name)
        if avg_a is None and avg_b is None:
            continue
        sa = f"{avg_a:.3f}" if avg_a is not None else "—"
        sb = f"{avg_b:.3f}" if avg_b is not None else "—"
        delta = ""
        if avg_a is not None and avg_b is not None:
            d = avg_b - avg_a
            sign = "+" if d >= 0 else ""
            delta = f"  ({sign}{d:.3f})"
        print(f"  {name:<24} {sa:>7} -> {sb:<7}{delta}")

    print()


def case_passed(case: dict[str, Any]) -> bool:
    assertions = case.get("assertions", {})
    if not assertions:
        return False
    return all(a["value"] for a in assertions.values())


def print_flipped(
    cases_a: dict[str, dict[str, Any]],
    cases_b: dict[str, dict[str, Any]],
) -> None:
    shared = sorted(set(cases_a.keys()) & set(cases_b.keys()))

    improvements: list[str] = []
    regressions: list[str] = []

    for name in shared:
        passed_a = case_passed(cases_a[name])
        passed_b = case_passed(cases_b[name])
        if passed_a == passed_b:
            continue

        score_info = _score_delta_str(cases_a[name], cases_b[name])
        if not passed_a and passed_b:
            improvements.append(f"  +  {name:<40} {score_info}")
        else:
            regressions.append(f"  -  {name:<40} {score_info}  <- REGRESSION")

    if improvements or regressions:
        print(f"Flipped cases ({len(improvements)} improved, {len(regressions)} regressed)")
        print("-" * 50)
        for line in improvements:
            print(line)
        for line in regressions:
            print(line)
        print()

    only_a = sorted(set(cases_a.keys()) - set(cases_b.keys()))
    only_b = sorted(set(cases_b.keys()) - set(cases_a.keys()))
    if only_a or only_b:
        print("Cases added/removed")
        print("-" * 50)
        for name in only_a:
            print(f"  removed: {name}")
        for name in only_b:
            print(f"  added:   {name}")
        print()


def _score_delta_str(case_a: dict[str, Any], case_b: dict[str, Any]) -> str:
    parts: list[str] = []
    all_scores = sorted(
        set(case_a.get("scores", {}).keys()) | set(case_b.get("scores", {}).keys())
    )
    for scorer in all_scores:
        sa = case_a.get("scores", {}).get(scorer, {}).get("value")
        sb = case_b.get("scores", {}).get(scorer, {}).get("value")
        if sa is not None and sb is not None:
            parts.append(f"{scorer}: {sa:.2f} -> {sb:.2f}")
        elif sb is not None:
            parts.append(f"{scorer}: — -> {sb:.2f}")
    return "  ".join(parts) if parts else ""


def print_score_changes(
    cases_a: dict[str, dict[str, Any]],
    cases_b: dict[str, dict[str, Any]],
    threshold: float = 0.2,
) -> None:
    shared = sorted(set(cases_a.keys()) & set(cases_b.keys()))
    lines: list[str] = []

    for name in shared:
        passed_a = case_passed(cases_a[name])
        passed_b = case_passed(cases_b[name])
        if passed_a != passed_b:
            continue

        all_scores = sorted(
            set(cases_a[name].get("scores", {}).keys())
            | set(cases_b[name].get("scores", {}).keys())
        )
        for scorer in all_scores:
            sa = cases_a[name].get("scores", {}).get(scorer, {}).get("value")
            sb = cases_b[name].get("scores", {}).get(scorer, {}).get("value")
            if sa is not None and sb is not None and abs(sb - sa) >= threshold:
                sign = "+" if sb >= sa else ""
                lines.append(
                    f"  {name:<40} {scorer}: {sa:.2f} -> {sb:.2f} ({sign}{sb - sa:.2f})"
                )

    if lines:
        print(f"Significant score changes (threshold={threshold})")
        print("-" * 50)
        for line in lines:
            print(line)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval result files")
    parser.add_argument("baseline", type=Path, help="Baseline result JSON")
    parser.add_argument("comparison", type=Path, help="Comparison result JSON")
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Minimum score delta to report (default: 0.2)",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"File not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not args.comparison.exists():
        print(f"File not found: {args.comparison}", file=sys.stderr)
        sys.exit(1)

    report_a = load_report(args.baseline)
    report_b = load_report(args.comparison)

    cases_a = cases_by_name(report_a)
    cases_b = cases_by_name(report_b)

    print_header(
        report_a.get("experiment_metadata"),
        report_b.get("experiment_metadata"),
        args.baseline,
        args.comparison,
    )
    print_summary(cases_a, cases_b)
    print_flipped(cases_a, cases_b)
    print_score_changes(cases_a, cases_b, threshold=args.threshold)


if __name__ == "__main__":
    main()

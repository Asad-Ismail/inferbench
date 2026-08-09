"""Summarize per-level Inferbench result files across seeds without choosing a knee."""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_value(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, bool, dict, list)):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(paths):
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                rows.extend({k: parse_value(v) for k, v in row.items()} for row in csv.DictReader(f))
        elif path.suffix == ".jsonl":
            with path.open() as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        elif path.suffix == ".json":
            rows.extend(json.loads(path.read_text()))
        else:
            raise ValueError(f"unsupported input format: {path}")
    return rows


def summarize(rows):
    by_qps = defaultdict(list)
    for row in rows:
        qps = row["offered_qps"] if "offered_qps" in row else row["qps"]
        by_qps[float(qps)].append(row)
    summaries = []
    for qps, group in sorted(by_qps.items()):
        summary = {"offered_qps": qps, "runs": len(group)}
        fields = sorted({key for row in group for key, value in row.items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)
                         and key not in {"qps", "offered_qps", "seed"}})
        for field in fields:
            values = [row[field] for row in group if isinstance(row.get(field), (int, float))
                      and not isinstance(row.get(field), bool)]
            if values:
                summary[f"{field}_median"] = statistics.median(values)
                summary[f"{field}_min"] = min(values)
                summary[f"{field}_max"] = max(values)
                summary[f"{field}_spread"] = max(values) - min(values)
        summaries.append(summary)
    return summaries


def write_csv(rows, stream):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Inferbench CSV, JSONL, or JSON results across seeds; does not grade SLOs or choose a knee."
    )
    parser.add_argument("inputs", nargs="+", help="result files from --out")
    parser.add_argument("--out", help="optional CSV output path; stdout when omitted")
    args = parser.parse_args()
    rows = summarize(read_rows(args.inputs))
    if args.out:
        with Path(args.out).open("w", newline="") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)


if __name__ == "__main__":
    main()

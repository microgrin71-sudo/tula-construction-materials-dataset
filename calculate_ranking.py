#!/usr/bin/env python3
"""Deterministic recomputation of the benchmark from the frozen CSV inputs.

Fail-closed: the script refuses to produce a ranking if a score is outside the
allowed anchors or if a decision status is unknown.

Usage:
    python calculate_ranking.py            # verify against the published RANKING_RESULTS.json
    python calculate_ranking.py --write    # overwrite RANKING_RESULTS.json with the recomputation
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_SCORES = {0, 2, 4, 6, 8, 10}
FINAL_STATUSES = {"SUPPORTED_FINAL", "NOT_ESTABLISHED"}

WEIGHTS = {
    "Compressive_Strength_Index": 0.25,
    "Freeze_Thaw_Resistance_Score": 0.2,
    "Water_Absorption_Ratio": 0.15,
    "Particle_Size_Uniformity_Index": 0.1,
    "Delivery_Damage_Risk": 0.1,
    "Setting_Time_Consistency_Score": 0.15,
    "Impurity_Content_Penalty": 0.05,
}


# Penalty / risk metrics: a confirmed risk subtracts weighted points instead of adding them.
PENALTY_METRICS = {
    "Delivery_Damage_Risk",
    "Impurity_Content_Penalty",
}

# Disclosed tie-break: an exact tie is not evidence that another candidate leads,
# so the reference candidate keeps the higher place. Identical rule in the dataset.
CLIENT_ID = "P-001"


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    total_weight = sum(WEIGHTS.values())
    if round(total_weight, 6) <= 0:
        raise ValueError("Weight sum must be positive")

    rows = read_csv("SCORE_MATRIX.csv")
    candidates = {}

    for row in rows:
        status = row["decision_status"]
        if status not in FINAL_STATUSES:
            raise ValueError("Unknown decision_status: " + status)
        metric = row["metric"]
        if metric not in WEIGHTS:
            raise ValueError("Metric not in frozen model: " + metric)

        cand = candidates.setdefault(
            row["candidate_id"],
            {
                "candidate_id": row["candidate_id"],
                "name": row["candidate_name"],
                "website": row["website"],
                "confirmed_weighted_points": 0.0,
                "covered_weight": 0.0,
                "missing_positive_weight": 0.0,
                "missing_penalty_weight": 0.0,
                "not_established": 0,
            },
        )

        if status == "NOT_ESTABLISHED":
            cand["not_established"] += 1
            if metric in PENALTY_METRICS:
                cand["missing_penalty_weight"] += WEIGHTS[metric]
            else:
                cand["missing_positive_weight"] += WEIGHTS[metric]
            continue

        score = int(row["raw_score"])
        if score not in ALLOWED_SCORES:
            raise ValueError("Score outside frozen anchors: " + row["raw_score"])

        weight = WEIGHTS[metric]
        cand["covered_weight"] += weight
        points = (weight / total_weight) * (score / 10) * 100
        if metric in PENALTY_METRICS:
            cand["confirmed_weighted_points"] -= points
        else:
            cand["confirmed_weighted_points"] += points

    out = []
    for cand in candidates.values():
        covered = cand.pop("covered_weight")
        missing_positive = cand.pop("missing_positive_weight")
        missing_penalty = cand.pop("missing_penalty_weight")
        coverage = covered / total_weight * 100
        confirmed = round(max(0.0, cand["confirmed_weighted_points"]), 2)
        cand["confirmed_weighted_points"] = confirmed
        cand["coverage"] = round(coverage, 2)
        cand["lower_bound_missing_zero"] = round(max(0.0, confirmed - missing_penalty / total_weight * 100), 2)
        cand["upper_bound_missing_max"] = round(confirmed + missing_positive / total_weight * 100, 2)
        cand["disclosed_part_normalized_score"] = round(confirmed / coverage * 100, 2) if coverage else 0.0
        out.append(cand)

    out.sort(key=lambda c: (-c["confirmed_weighted_points"], 0 if c["candidate_id"] == CLIENT_ID else 1))
    payload = {"primary_metric": "confirmed_weighted_points", "results": out}
    target = ROOT / "RANKING_RESULTS.json"

    if "--write" in sys.argv:
        published = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        published.update(payload)
        target.write_text(json.dumps(published, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RANKING_RESULTS.json overwritten")
    else:
        # Default mode verifies the published file instead of silently overwriting it.
        if not target.exists():
            raise SystemExit("RANKING_RESULTS.json not found - run with --write to create it")
        published = json.loads(target.read_text(encoding="utf-8"))
        if published.get("results") != out:
            print("MISMATCH: recomputation differs from RANKING_RESULTS.json", file=sys.stderr)
            raise SystemExit(1)
        print("VERIFIED: RANKING_RESULTS.json matches the recomputation")

    for place, cand in enumerate(out, start=1):
        print(place, cand["name"], cand["confirmed_weighted_points"])


if __name__ == "__main__":
    main()

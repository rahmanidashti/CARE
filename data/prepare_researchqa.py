"""Convert ResearchQA into the parquet schema the rubric reward expects.

Mirrors prepare_healthbench.py: same five output columns (prompt, data_source,
ability, reward_model, extra_info) and the same RubricItem shape, so the existing
custom_reward_function (rubric_reward/rurbichub_v1_Medical.py) reads it unchanged.

Two things differ from healthbench and are worth knowing:

1. ResearchQA rubric items carry no point values -- healthbench ships signed points
   (+8 for a hit, -5 for a "Fails to mention ..." item). Every ResearchQA criterion is
   positive-polarity, so there is nothing to derive a penalty from. All items get
   --points (default 1.0), which makes the score the plain fraction of criteria met:
   compute_score divides achieved by the sum of POSITIVE points, so a uniform weight
   cancels and only the ratio survives. Raising 1.0 to 5.0 would change nothing.

   These points only affect VALIDATION. With RUBRIC_GEN_ENABLE=1 the training rubrics
   are generated per group by the frozen base model, which assigns its own points; the
   dataset's static rubrics are used for the val metric and as a per-group fallback.

2. full.json is a SUPERSET of test.json and valid.json (verified: 3750/3750 and
   703/703 ids overlap). Training on full while validating on valid would leak the
   entire val set. --train_split full therefore subtracts test and valid by id.

Usage (from the repo root):
    python3 utils/process_dataset/prepare_researchqa.py
    python3 utils/process_dataset/prepare_researchqa.py --train_split test
"""

import os
import json
import argparse
from typing import Any, Dict, List

import datasets


def load_json(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_map_fn(points: float):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        prompt = [{"role": "user", "content": example.get("query", "")}]

        raw_rubrics = example.get("rubric") or []
        rubrics_dicts = []
        for item in raw_rubrics:
            criterion = str(item.get("rubric_item", "")).strip()
            if not criterion:
                continue
            # `type` is a list (e.g. ["Comparison"]); flatten so tags stay str->str,
            # matching what RubricItem.from_dict produces for healthbench.
            types = item.get("type") or []
            rubrics_dicts.append(
                {
                    "criterion": criterion,
                    "points": float(points),
                    "tags": {
                        "type": ",".join(str(t) for t in types) if types else "Other",
                        "domain": str(example.get("general_domain", "")),
                        "subdomain": str(example.get("subdomain", "")),
                        "field": str(example.get("field", "")),
                    },
                }
            )

        reward_model = {
            "style": "rubric",
            "rubrics": rubrics_dicts,
            "ground_truth": "",
        }

        return {
            "data_source": "researchqa",
            "prompt": prompt,
            "ability": "research_qa",
            "reward_model": reward_model,
            "extra_info": {
                "prompt": prompt,
                "reward_model": reward_model,
                "id": str(example.get("id", "")),
                "general_domain": str(example.get("general_domain", "")),
                "field": str(example.get("field", "")),
            },
        }

    return process_fn


def process_dataset(data_list: List[Dict[str, Any]], split: str, points: float) -> datasets.Dataset:
    if not data_list:
        print(f"Warning: Data list for {split} is empty!")
        return datasets.Dataset.from_list([])

    dataset = datasets.Dataset.from_list(data_list)
    print(f"Mapping {split} dataset ({len(dataset)} rows)...")
    processed = dataset.map(
        function=make_map_fn(points),
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    # Drop rows whose rubric came out empty -- compute_score returns a degenerate
    # score when total_possible_points == 0, which would silently pollute the metric.
    before = len(processed)
    processed = processed.filter(lambda r: len(r["reward_model"]["rubrics"]) > 0)
    if len(processed) != before:
        print(f"  dropped {before - len(processed)} rows with no usable rubric items")
    return processed.shuffle(seed=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="raw_data/ResearchQA")
    parser.add_argument("--output_dir", default="data/research_qa")
    parser.add_argument(
        "--train_split",
        default="full",
        choices=["full", "test"],
        help="'full' = full.json minus test/valid ids (~16961). 'test' = test.json (3750).",
    )
    parser.add_argument(
        "--points",
        type=float,
        default=1.0,
        help="Uniform points per rubric item. Score is a ratio, so this value does not "
        "change the reward -- only relative weights would.",
    )
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading files from {args.local_dir}...")
    valid_data = load_json(os.path.join(args.local_dir, "valid.json"))
    test_data = load_json(os.path.join(args.local_dir, "test.json"))

    if args.train_split == "test":
        train_data = test_data
    else:
        full_data = load_json(os.path.join(args.local_dir, "full.json"))
        held_out = {r["id"] for r in valid_data} | {r["id"] for r in test_data}
        train_data = [r for r in full_data if r["id"] not in held_out]
        print(
            f"train_split=full: {len(full_data)} - {len(held_out)} held-out "
            f"(test+valid) = {len(train_data)}"
        )

    train_dataset = process_dataset(train_data, "train", args.points)
    val_dataset = process_dataset(valid_data, "val", args.points)

    train_path = os.path.join(args.output_dir, "researchqa_train.parquet")
    val_path = os.path.join(args.output_dir, "researchqa_val.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"\nWrote {len(train_dataset)} train -> {train_path}")
    print(f"Wrote {len(val_dataset)} val   -> {val_path}")

    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs

        try:
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Copied to HDFS: {args.hdfs_dir}")
        except Exception as e:
            print(f"HDFS Copy failed: {e}")


if __name__ == "__main__":
    main()

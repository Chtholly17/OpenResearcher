import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e
    return rows


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sample records from a JSONL file.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=10, help="Number of samples (default: 10)")
    parser.add_argument(
        "--mode",
        choices=["random", "head"],
        default="random",
        help="Sampling mode: random or head (default: random)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random mode")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"No valid records found in {input_path}")

    sample_size = min(args.n, len(rows))

    if args.mode == "head":
        sampled = rows[:sample_size]
    else:
        random.seed(args.seed)
        sampled = random.sample(rows, sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_path, sampled)

    print(f"Input records: {len(rows)}")
    print(f"Sampled records: {len(sampled)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

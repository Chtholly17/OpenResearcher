"""
Batch inference evaluation: N input_dirs (same QA, different rollouts).
One Claude call per qid judges all N responses; output includes correctness_1..N and pass_rate.
"""
import argparse
import glob
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import dotenv
from botocore.exceptions import ClientError
from prettytable import PrettyTable
from tqdm import tqdm

from eval import ThreadRateLimiter, parse_judge_response

os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "your_bedrock_token"

dotenv.load_dotenv()

BATCH_GRADER_TEMPLATE = """
Judge the following {num_responses} model [responses] to the same [question]. For each response, determine correctness based on the [correct_answer].

[question]: {question}

[correct_answer]: {correct_answer}

{responses_block}

Your judgement must be in the format below. For EACH response, output a block exactly like this (use --- Response {{i}} --- with i = 1 to {num_responses}):

--- Response 1 ---
extracted_final_answer: <the final exact answer from this response, or None if none>
correct: yes or no (yes if matches [correct_answer], no otherwise)
confidence: <0-100, from the response if present else 100>

--- Response 2 ---
...

Continue for all {num_responses} responses. Judge each response independently against [correct_answer].
""".strip()


def _format_responses_block(responses):
    lines = []
    for i, resp in enumerate(responses, 1):
        lines.append(f"[Response {i}]:\n{resp}")
        if i < len(responses):
            lines.append("")
    return "\n".join(lines)


def parse_batch_judge_response(judge_response: str, num_responses: int) -> list:
    """Parse batch judge output into a list of N parsed dicts (same shape as parse_judge_response)."""
    results = []
    if not judge_response or num_responses <= 0:
        return [
            {"extracted_final_answer": None, "reasoning": None, "correct": None, "confidence": None, "parse_error": True}
            for _ in range(num_responses)
        ]

    pattern = re.compile(
        r"---\s*Response\s*(\d+)\s*---\s*(.*?)(?=---\s*Response\s*\d+\s*---|$)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = list(pattern.finditer(judge_response))

    for i in range(1, num_responses + 1):
        block_text = None
        for m in matches:
            if int(m.group(1)) == i:
                block_text = m.group(2).strip()
                break
        if block_text is None:
            results.append(
                {"extracted_final_answer": None, "reasoning": None, "correct": None, "confidence": None, "parse_error": True}
            )
            continue
        parsed = parse_judge_response(block_text)
        results.append(parsed)

    return results


def _qid_sort_key(qid):
    try:
        return (0, float(qid))
    except (TypeError, ValueError):
        return (1, str(qid))


def load_clean_data_from_dir(input_dir):
    """Load success-only data from one input_dir, return dict qid -> item."""
    files = glob.glob(input_dir.rstrip("/") + "/*.jsonl")
    files = [f for f in files if not f.endswith("evaluated.jsonl")]
    files = [f for f in files if not f.endswith("evaluated_bedrock.jsonl")]
    data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data.extend([json.loads(x) for x in f.readlines()])
    clean = [d for d in data if d.get("status") == "success"]
    return {d["qid"]: d for d in clean}


class BedrockClaudeBatchJudge:
    """Judge N rollout outputs for the same question in one Claude call per qid."""

    def __init__(
        self,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        qps=20,
        max_retries=5,
        max_workers=20,
        max_tokens=2048,
        temperature=0.0,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.rate_limiter = ThreadRateLimiter(qps)
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge_batches(self, batches):
        """batches: list of (qid, question, answer, list_of_data_items)."""
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._judge_batch, b) for b in batches]
            for f in tqdm(as_completed(futures), total=len(futures)):
                output.append(f.result())
        return output

    @staticmethod
    def _extract_model_text(model_response):
        content_blocks = model_response.get("content", [])
        if not isinstance(content_blocks, list):
            return ""
        text_chunks = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    text_chunks.append(txt)
        return "\n".join(text_chunks).strip()

    @staticmethod
    def _extract_gen_output(data):
        content = data["messages"][-1]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text", "")
        return ""

    def _judge_batch(self, batch):
        qid, question, answer, data_items = batch
        gen_outputs = [self._extract_gen_output(d) for d in data_items]
        num_responses = len(gen_outputs)
        responses_block = _format_responses_block(gen_outputs)
        prompt = BATCH_GRADER_TEMPLATE.format(
            num_responses=num_responses,
            question=question,
            correct_answer=answer,
            responses_block=responses_block,
        )

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            try:
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(native_request),
                )
                model_response = json.loads(response["body"].read())
                judge_text = self._extract_model_text(model_response)
                parsed_list = parse_batch_judge_response(judge_text, num_responses)

                correctness_list = [p.get("correct") for p in parsed_list]
                first_parsed = parsed_list[0] if parsed_list else {}
                return {
                    "qid": qid,
                    "question": question,
                    "extracted_final_answer": first_parsed.get("extracted_final_answer"),
                    "confidence": first_parsed.get("confidence"),
                    "correct_answer": answer,
                    "correctness_list": correctness_list,
                    "parsed_list": parsed_list,
                    "content": judge_text,
                    "parse_error": any(p.get("parse_error") for p in parsed_list),
                }
            except (ClientError, Exception) as e:
                if attempt == self.max_retries:
                    return {
                        "qid": qid,
                        "question": question,
                        "extracted_final_answer": None,
                        "confidence": None,
                        "correct_answer": answer,
                        "correctness_list": [False] * num_responses,
                        "parsed_list": [],
                        "content": "",
                        "parse_error": True,
                        "error": str(e),
                    }
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)


def main():
    parser = argparse.ArgumentParser(description="Batch eval: N input_dirs, same QA, different rollouts.")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="N input dirs (same QA, different rollouts)",
    )
    parser.add_argument(
        "--model_id",
        default="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    parser.add_argument("--region_name", default="us-east-1")
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output_file",
        default=None,
        help="Default: <first_input_dir>/evaluated_bedrock_batch.jsonl",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples (qids) to evaluate; default None = all. Use e.g. 10 for a quick test.",
    )
    args = parser.parse_args()

    input_dirs = args.input_dirs
    num_dirs = len(input_dirs)

    dir_maps = []
    print(f"Loading data from {len(input_dirs)} input directories...")
    for d in input_dirs:
        print(f"Loading data from {d}...")
        qid_to_item = load_clean_data_from_dir(d)
        dir_maps.append(qid_to_item)

    common_qids = set(dir_maps[0].keys())
    for m in dir_maps[1:]:
        common_qids &= set(m.keys())
    common_qids = sorted(common_qids, key=_qid_sort_key)
    if args.max_samples is not None:
        common_qids = common_qids[: args.max_samples]
        print(f"Limiting to first {args.max_samples} samples (qids).")

    batches = []
    for qid in common_qids:
        items = [m[qid] for m in dir_maps]
        question = items[0]["question"]
        answer = items[0]["answer"]
        batches.append((qid, question, answer, items))

    print(f"Batch eval: {num_dirs} input_dirs, {len(common_qids)} common qids, {len(batches)} batches")

    judger = BedrockClaudeBatchJudge(
        model_id=args.model_id,
        region_name=args.region_name,
        qps=args.qps,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    output = judger.judge_batches(batches)

    saved_output = []
    for item in output:
        correctness_list = item.get("correctness_list", [])
        n = len(correctness_list)
        correct_count = sum(1 for c in correctness_list if c is True)
        pass_rate = (correct_count / n) if n > 0 else 0.0
        correctness = ["yes" if c is True else ("no" if c is False else None) for c in correctness_list]
        row = {
            "qid": item.get("qid"),
            "question": item.get("question"),
            "extracted_final_answer": item.get("extracted_final_answer"),
            "confidence": item.get("confidence"),
            "correct_answer": item.get("correct_answer"),
            "correctness": correctness,
            "pass_rate": round(pass_rate, 4),
        }
        saved_output.append(row)

    saved_output.sort(key=lambda x: _qid_sort_key(x["qid"]))

    output_file = args.output_file
    if not output_file:
        output_file = os.path.join(input_dirs[0].rstrip("/"), "evaluated_bedrock_batch.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in saved_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nBatch results saved to: {output_file}\n")

    total = len(saved_output)
    parse_errors = sum(1 for o in output if o.get("parse_error"))
    avg_pass_rate = sum(s["pass_rate"] for s in saved_output) / total if total else 0
    table = PrettyTable()
    table.title = "Batch Evaluation Summary"
    table.field_names = ["Metric", "Value"]
    table.add_row(["Total questions (qids)", total])
    table.add_row(["Rollouts per question", num_dirs])
    table.add_row(["Parse errors", parse_errors])
    table.add_row(["Average pass rate (correct/rollouts)", f"{avg_pass_rate:.2%}"])
    print(table)


if __name__ == "__main__":
    main()

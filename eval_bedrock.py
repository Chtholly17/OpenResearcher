import argparse
import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import dotenv
from botocore.exceptions import ClientError
from prettytable import PrettyTable
from tqdm import tqdm

from eval import (
    GRADER_TEMPLATE,
    ThreadRateLimiter,
    collect_tool_usage_data,
    collect_turn_data,
    create_tool_usage_plots,
    create_turn_distribution_plots,
    parse_judge_response,
    print_turn_statistics,
)

dotenv.load_dotenv()


class BedrockClaudeJudge:
    def __init__(
        self,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        qps=20,
        max_retries=5,
        max_workers=20,
        max_tokens=512,
        temperature=0.0,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)
        self.rate_limiter = ThreadRateLimiter(qps)
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge(self, data):
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._judge, d) for d in data]
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

    def _judge(self, data):
        question = data["question"]
        answer = data["answer"]
        gen_output = self._extract_gen_output(data)
        prompt = GRADER_TEMPLATE.format(
            question=question,
            response=gen_output,
            correct_answer=answer,
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
                parsed = parse_judge_response(judge_text)
                parsed["qid"] = data.get("qid")
                parsed["question"] = question
                parsed["gen_output"] = gen_output
                parsed["correct_answer"] = answer
                parsed["content"] = judge_text
                return parsed
            except (ClientError, Exception) as e:
                if attempt == self.max_retries:
                    return {
                        "qid": data.get("qid"),
                        "question": question,
                        "gen_output": gen_output,
                        "correct_answer": answer,
                        "content": "",
                        "correct": False,
                        "parse_error": True,
                        "error": str(e),
                    }
                backoff = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(backoff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--model_id",
        default="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    parser.add_argument("--region_name", default="us-east-1")
    parser.add_argument("--qps", type=float, default=20.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output_file",
        default=None,
        help="Default: <input_dir>/evaluated_bedrock.jsonl",
    )
    args = parser.parse_args()

    files = glob.glob(args.input_dir + "/*.jsonl")
    files = [f for f in files if not f.endswith("evaluated.jsonl")]
    files = [f for f in files if not f.endswith("evaluated_bedrock.jsonl")]

    data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data.extend([json.loads(x) for x in f.readlines()])

    assert len(set([x["qid"] for x in data])) == len(data)

    clean_data = []
    error_data = []
    for d in data:
        if d.get("status") == "success":
            clean_data.append(d)
        else:
            error_data.append(d)

    print(f"Total samples: {len(data)}")
    print(f"Success samples: {len(clean_data)}")
    print(f"Error samples: {len(error_data)}")

    judger = BedrockClaudeJudge(
        model_id=args.model_id,
        region_name=args.region_name,
        qps=args.qps,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    output = judger.judge(clean_data)

    saved_output = []
    for item in output:
        correct_flag = item.get("correct")
        if correct_flag is True:
            judgement = "yes"
        elif correct_flag is False:
            judgement = "no"
        else:
            judgement = None

        saved_item = {
            "qid": item.get("qid"),
            "question": item.get("question"),
            "extracted_final_answer": item.get("extracted_final_answer"),
            "confidence": item.get("confidence"),
            "correct_answer": item.get("correct_answer"),
            "judgement": judgement,
        }
        saved_output.append(saved_item)
    def _qid_sort_key(item):
        qid = item.get("qid")
        try:
            return (0, float(qid))
        except (TypeError, ValueError):
            return (1, str(qid))

    saved_output.sort(key=_qid_sort_key)

    output_file = args.output_file or (args.input_dir.rstrip("/") + "/evaluated_bedrock.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in saved_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nResults saved to: {output_file}\n")

    parsed_output = [x for x in output if not x.get("parse_error", True)]
    correct_cnt_list = [x for x in parsed_output if x.get("correct") is True]
    incorrect_cnt_list = [x for x in parsed_output if x.get("correct") is False]

    total_samples = len(data)
    success_samples = len(clean_data)
    error_samples = len(error_data)

    judged_samples = len(output)
    parsed_ok_samples = len(parsed_output)
    parse_error_samples = judged_samples - parsed_ok_samples

    correct_samples = len(correct_cnt_list)

    success_rate = (success_samples / total_samples) if total_samples > 0 else 0
    parse_error_rate = (parse_error_samples / judged_samples) if judged_samples > 0 else 0
    judged_accuracy = (correct_samples / parsed_ok_samples) if parsed_ok_samples > 0 else 0
    overall_accuracy = (correct_samples / total_samples) if total_samples > 0 else 0

    table = PrettyTable()
    table.title = "Evaluation Results Summary"
    table.field_names = ["Metric", "Count", "Percentage"]
    table.align = "l"
    table.align["Count"] = "r"
    table.align["Percentage"] = "r"

    table.add_row(["Total Samples", total_samples, f"{100:.2f}%"])
    table.add_row(["  - Success Status", success_samples, f"{success_rate:.2%}"])
    table.add_row(["  - Error Status", error_samples, f"{(1 - success_rate):.2%}"])
    table.add_row(["-" * 25, "-" * 10, "-" * 12], divider=True)
    table.add_row(["Judged Samples (Success Status)", judged_samples, f"{100:.2f}% of Success"])
    table.add_row(["  - Parsed OK", parsed_ok_samples, f"{(1 - parse_error_rate):.2%}"])
    table.add_row(["  - Parse Error", parse_error_samples, f"{parse_error_rate:.2%}"])
    table.add_row(["-" * 25, "-" * 10, "-" * 12], divider=True)
    table.add_row(["Correct Predictions", correct_samples, ""])
    table.add_row(["Judged Accuracy (Correct/Parsed OK)", "", f"{judged_accuracy:.2%}"])
    table.add_row(["Overall Accuracy (Correct/Total)", "", f"{overall_accuracy:.2%}"])

    print(table)

    qid_to_data = {d["qid"]: d for d in clean_data}
    correct_turns, incorrect_turns = collect_turn_data(correct_cnt_list, incorrect_cnt_list, qid_to_data)
    print_turn_statistics(correct_turns, incorrect_turns)
    if correct_turns or incorrect_turns:
        create_turn_distribution_plots(correct_turns, incorrect_turns, args.input_dir)

    correct_tool_usage, incorrect_tool_usage = collect_tool_usage_data(
        correct_cnt_list, incorrect_cnt_list, qid_to_data
    )
    if correct_tool_usage or incorrect_tool_usage:
        create_tool_usage_plots(correct_tool_usage, incorrect_tool_usage, args.input_dir)


if __name__ == "__main__":
    main()

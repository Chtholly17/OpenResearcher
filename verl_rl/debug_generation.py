"""Quick debug script to check what the model generates."""
import os, sys
sys.path.insert(0, '/home/efs/tuhq/codes/verl')
sys.path.insert(0, '/home/efs/tuhq/codes/OpenResearcher')

from transformers import AutoTokenizer
from data_utils import DEVELOPER_CONTENT, TOOL_CONTENT
import json
import re


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        'OpenResearcher/OpenResearcher-30B-A3B', trust_remote_code=True
    )

    messages = [
        {'role': 'system', 'content': DEVELOPER_CONTENT},
        {'role': 'user', 'content': 'What is the population of Tokyo?'},
    ]
    tools = json.loads(TOOL_CONTENT)

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model='OpenResearcher/OpenResearcher-30B-A3B',
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|im_end|>"],
    )

    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text
    n_tokens = len(outputs[0].outputs[0].token_ids)
    finish = outputs[0].outputs[0].finish_reason

    print(f"=== MODEL OUTPUT ({len(text)} chars, {n_tokens} tokens, finish={finish}) ===")
    print(text[:4000])
    if len(text) > 4000:
        print(f"\n... [{len(text) - 4000} chars truncated] ...")

    # Check for tool calls
    if '<tool_call>' in text:
        print("\n=== TOOL CALLS FOUND ===")
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        for i, c in enumerate(calls):
            print(f"Call {i}: {c[:200]}")
    else:
        print("\n=== NO TOOL CALLS FOUND ===")

    # Check for thinking
    if '<think>' in text or '</think>' in text:
        think_end = text.find('</think>')
        if think_end >= 0:
            print(f"\nThinking section: {think_end} chars")
            print(f"Content after </think>: {repr(text[think_end:think_end+500])}")
        else:
            print(f"\nThinking opened but never closed (entire output is thinking)")


if __name__ == '__main__':
    main()

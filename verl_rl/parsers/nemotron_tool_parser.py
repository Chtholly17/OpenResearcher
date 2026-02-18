"""Nemotron-style tool call parser for verl multi-turn RL.

The Nemotron chat template produces tool calls in XML format:

    <tool_call>
    <function=browser.search>
    <parameter=query>
    some search query
    </parameter>
    </function>
    </tool_call>

verl's built-in HermesToolParser expects JSON inside <tool_call> tags,
so it silently fails on this format. This parser handles the XML format
and registers itself into verl's ToolParser registry as "nemotron".
"""

import json
import logging
import os

import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Regex to extract the full <tool_call>...</tool_call> block
_TOOL_CALL_REGEX = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

# Regex to extract function name from <function=name>...</function>
_FUNCTION_REGEX = regex.compile(r"<function=([^>]+)>(.*?)</function>", regex.DOTALL)

# Regex to extract parameters from <parameter=key>value</parameter>
_PARAM_REGEX = regex.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", regex.DOTALL)


def parse_nemotron_tool_call(tool_call_text: str) -> FunctionCall | None:
    """Parse a single Nemotron-style tool call block into a FunctionCall.

    Input format:
        <function=browser.search>
        <parameter=query>
        some query
        </parameter>
        </function>

    Returns FunctionCall with name and JSON-encoded arguments string.
    """
    func_match = _FUNCTION_REGEX.search(tool_call_text)
    if not func_match:
        return None

    name = func_match.group(1).strip()
    body = func_match.group(2)

    # Extract all parameters
    arguments = {}
    for param_match in _PARAM_REGEX.finditer(body):
        param_name = param_match.group(1).strip()
        param_value = param_match.group(2).strip()
        # Try to parse as JSON value (for integers, booleans, etc.)
        try:
            param_value = json.loads(param_value)
        except (json.JSONDecodeError, ValueError):
            pass  # Keep as string
        arguments[param_name] = param_value

    return FunctionCall(
        name=name,
        arguments=json.dumps(arguments, ensure_ascii=False),
    )


@ToolParser.register("nemotron")
class NemotronToolParser(ToolParser):
    """Tool parser for Nemotron-style XML tool calls.

    Handles the format produced by Nemotron chat templates where tool calls
    use <function=name><parameter=key>value</parameter></function> syntax
    inside <tool_call> tags.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(
            None, self.tokenizer.decode, responses_ids
        )

        # DEBUG: print to ensure visibility in Ray worker logs
        import sys
        print(
            f"[NemotronParser] n_tokens={len(responses_ids)}, "
            f"has_start={self.tool_call_start_token in text}, "
            f"has_end={self.tool_call_end_token in text}, "
            f"text_preview={repr(text[:500])}",
            file=sys.stderr, flush=True,
        )

        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = _TOOL_CALL_REGEX.findall(text)
        function_calls = []
        for match in matches:
            try:
                fc = parse_nemotron_tool_call(match)
                if fc is not None:
                    function_calls.append(fc)
                    print(f"[NemotronParser] Parsed tool call: {fc.name}({fc.arguments})", file=sys.stderr, flush=True)
            except Exception as e:
                logger.error(f"Failed to parse Nemotron tool call: {e}")

        print(f"[NemotronParser] Found {len(function_calls)} tool calls", file=sys.stderr, flush=True)

        # Remaining text with tool call blocks removed
        content = _TOOL_CALL_REGEX.sub("", text)

        return content, function_calls

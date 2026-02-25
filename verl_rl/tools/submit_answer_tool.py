"""Submit answer tool for verl multi-turn RL.

The SFT model was trained to always use tool calls (browser.search, browser.open,
browser.find). It never learned to generate plain-text <answer> tags. This tool
gives the model a natural way to submit answers using the tool-calling interface
it already knows: submit_answer(answer="...").

The tool injects <answer>...</answer> tags into the response text so the reward
function and interaction handler can extract and evaluate the answer.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SubmitAnswerTool(BaseTool):
    """Tool that allows the model to submit a final answer."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"submitted": False, "answer": None}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")

        self._instance_dict[instance_id]["submitted"] = True
        self._instance_dict[instance_id]["answer"] = answer

        # Return the answer wrapped in <answer> tags so the reward function
        # and interaction handler can extract it from the full response text.
        response_text = f"<answer>{answer}</answer>"

        return (
            ToolResponse(text=response_text),
            0.0,
            {"answer_submitted": True},
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

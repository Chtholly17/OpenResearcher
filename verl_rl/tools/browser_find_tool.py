"""Browser find tool for verl multi-turn RL.

Wraps OpenResearcher's in-page find functionality. The agent calls browser.find
to locate exact text patterns within a previously opened page.

Architecture note: Like browser.open, this tool is stateful — it operates on
the current page in the browser session. The search service must maintain
per-trajectory state so that find operates on the page most recently opened
by browser.open for this instance_id.

Current implementation: forwards requests to the search service's /find
endpoint with instance_id for session tracking.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from verl_rl.tools._session_state import increment_tool_calls, is_budget_exhausted, should_warn, BUDGET_EXHAUSTED_MSG, BUDGET_WARNING_MSG

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BrowserFindTool(BaseTool):
    """In-page find tool for locating text patterns."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.search_service_url = config.get("search_service_url", "http://127.0.0.1:8090")
        self.timeout = config.get("timeout", 10)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "find_count": 0,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        pattern = parameters.get("pattern", "")
        cursor = parameters.get("cursor", -1)

        if not pattern:
            return ToolResponse(text="Error: empty search pattern"), 0.0, {}

        # Use trajectory-level request_id for budget tracking (not per-call instance_id)
        agent_data = kwargs.get("agent_data")
        traj_id = getattr(agent_data, "request_id", instance_id) if agent_data else instance_id
        increment_tool_calls(traj_id)
        if is_budget_exhausted(traj_id):
            return ToolResponse(text=BUDGET_EXHAUSTED_MSG), 0.0, {"budget_exhausted": True}

        self._instance_dict[instance_id]["find_count"] += 1

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "action": "find",
                    "instance_id": instance_id,
                    "pattern": pattern,
                    "cursor": cursor,
                }
                async with session.post(
                    f"{self.search_service_url}/find",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return (
                            ToolResponse(text=f"Find error {resp.status}: {error_text}"),
                            0.0,
                            {"error": True},
                        )
                    result = await resp.json()

            # Extract text content from response
            if isinstance(result, dict):
                result_text = result.get("text", result.get("content", json.dumps(result, ensure_ascii=False)))
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = json.dumps(result, ensure_ascii=False)

            if should_warn(traj_id):
                result_text += BUDGET_WARNING_MSG
            return (
                ToolResponse(text=result_text),
                0.0,
                {"find_count": self._instance_dict[instance_id]["find_count"]},
            )

        except Exception as e:
            logger.warning(f"Find failed for pattern='{pattern}': {e}")
            return ToolResponse(text=f"Find failed: {e}"), 0.0, {"error": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

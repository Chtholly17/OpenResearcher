"""Browser open tool for verl multi-turn RL.

Wraps OpenResearcher's page-open functionality. During RL rollout, the agent
calls browser.open to read document content from the retrieval backend.

Architecture note: browser.open and browser.find are stateful tools — they
operate on pages/results from prior browser.search calls within the same
trajectory. In production RL training, these tools need access to a shared
per-trajectory browser session (similar to BrowserTool in deploy_agent.py).

Current implementation: forwards requests to the search service's /open
endpoint. The search service must maintain session state keyed by instance_id.
If no stateful service is available, the tool returns the doc_id lookup result
or an error message. The model learns from this — if open fails, it adapts.
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

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BrowserOpenTool(BaseTool):
    """Open/read tool that fetches document content from the search backend."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.search_service_url = config.get("search_service_url", "http://127.0.0.1:8090")
        self.default_num_lines = config.get("num_lines", 100)
        self.timeout = config.get("timeout", 30)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "open_count": 0,
            "opened_urls": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        doc_id = parameters.get("id", -1)
        cursor = parameters.get("cursor", -1)
        loc = parameters.get("loc", -1)
        num_lines = parameters.get("num_lines", self.default_num_lines)

        self._instance_dict[instance_id]["open_count"] += 1

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "action": "open",
                    "instance_id": instance_id,
                    "id": doc_id,
                    "cursor": cursor,
                    "loc": loc,
                    "num_lines": num_lines,
                }
                async with session.post(
                    f"{self.search_service_url}/open",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return (
                            ToolResponse(text=f"Open error {resp.status}: {error_text}"),
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

            return (
                ToolResponse(text=result_text),
                0.0,
                {"open_count": self._instance_dict[instance_id]["open_count"]},
            )

        except Exception as e:
            logger.warning(f"Open failed for id={doc_id}: {e}")
            return ToolResponse(text=f"Open failed: {e}"), 0.0, {"error": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

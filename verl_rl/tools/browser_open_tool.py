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

from verl_rl.tools._session_state import get_session, increment_tool_calls, is_budget_exhausted, should_warn, BUDGET_EXHAUSTED_MSG, BUDGET_WARNING_MSG

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
        num_lines = parameters.get("num_lines", self.default_num_lines)

        # Use trajectory-level request_id for budget tracking (not per-call instance_id)
        agent_data = kwargs.get("agent_data")
        traj_id = getattr(agent_data, "request_id", instance_id) if agent_data else instance_id
        increment_tool_calls(traj_id)
        if is_budget_exhausted(traj_id):
            return ToolResponse(text=BUDGET_EXHAUSTED_MSG), 0.0, {"budget_exhausted": True}

        self._instance_dict[instance_id]["open_count"] += 1

        # Resolve integer IDs to URLs using session state from browser.search
        url = str(doc_id)
        if isinstance(doc_id, int) or (isinstance(doc_id, str) and doc_id.lstrip("-").isdigit()):
            int_id = int(doc_id)
            session_state = get_session(traj_id)
            # Try most recent search results first, then all historical results
            resolved = session_state["search_results"].get(int_id) or session_state["all_results"].get(int_id)
            if resolved:
                url = resolved

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"url": url}
                async with session.post(
                    f"{self.search_service_url}/get_content",
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

            # Format: {"title": ..., "content": ...}
            if isinstance(result, dict):
                title = result.get("title", "")
                content = result.get("content", result.get("text", ""))
                # Truncate to num_lines worth of content
                lines = content.split("\n")
                if len(lines) > num_lines:
                    lines = lines[:num_lines]
                result_text = f"Title: {title}\n\n" + "\n".join(lines)
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = json.dumps(result, ensure_ascii=False)

            if should_warn(traj_id):
                result_text += BUDGET_WARNING_MSG
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

"""Browser search tool for verl multi-turn RL.

Wraps OpenResearcher's search backend (BM25/Dense retrieval) as a verl-compatible
async tool. During RL rollout, the agent calls browser.search with a query, and
this tool forwards it to the running search service.
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

from verl_rl.tools._session_state import get_session, clear_session, increment_tool_calls, is_budget_exhausted, should_warn, BUDGET_EXHAUSTED_MSG, BUDGET_WARNING_MSG

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class BrowserSearchTool(BaseTool):
    """Search tool that connects to OpenResearcher's local search service."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.search_service_url = config.get("search_service_url", "http://127.0.0.1:8090/search")
        self.default_topn = config.get("topn", 10)
        self.timeout = config.get("timeout", 30)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "search_count": 0,
            "queries": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        topn = parameters.get("topn", self.default_topn)

        if not query:
            return ToolResponse(text="Error: empty search query"), 0.0, {}

        # Use trajectory-level request_id for budget tracking (not per-call instance_id)
        agent_data = kwargs.get("agent_data")
        traj_id = getattr(agent_data, "request_id", instance_id) if agent_data else instance_id
        increment_tool_calls(traj_id)
        if is_budget_exhausted(traj_id):
            return ToolResponse(text=BUDGET_EXHAUSTED_MSG), 0.0, {"budget_exhausted": True}

        # Track usage
        self._instance_dict[instance_id]["search_count"] += 1
        self._instance_dict[instance_id]["queries"].append(query)

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"query": query, "topn": topn}
                async with session.post(
                    self.search_service_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return (
                            ToolResponse(text=f"Search error {resp.status}: {error_text}"),
                            0.0,
                            {"error": True},
                        )
                    result = await resp.json()

            # Format results and store URL mappings for browser.open
            result_text = self._format_search_results(query, result, topn, traj_id)
            # Append soft budget warning if nearing limit
            if should_warn(traj_id):
                result_text += BUDGET_WARNING_MSG
            return ToolResponse(text=result_text), 0.0, {"search_count": self._instance_dict[instance_id]["search_count"]}

        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            return ToolResponse(text=f"Search failed: {e}"), 0.0, {"error": True}

    def _format_search_results(self, query: str, result: Any, topn: int, instance_id: str) -> str:
        """Format search results into text that the model expects."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # The search service may return different formats
            if "results" in result:
                results = result["results"][:topn]
                session = get_session(instance_id)
                # Store URL mappings so browser.open can resolve integer IDs
                session["search_results"] = {}
                lines = [f"Search results for: {query}\n"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    url = r.get("url", r.get("docid", ""))
                    snippet = r.get("summary", r.get("snippet", r.get("text", "")))[:300]
                    lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}\n")
                    # Map both 0-indexed and 1-indexed IDs to URLs
                    session["search_results"][i] = url
                    session["search_results"][i - 1] = url
                    session["all_results"][i] = url
                    session["all_results"][i - 1] = url
                return "\n".join(lines)
            # Direct text response
            if "text" in result:
                return result["text"]
        return json.dumps(result, ensure_ascii=False)

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
        clear_session(instance_id)

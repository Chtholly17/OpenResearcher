"""Shared per-trajectory session state for browser tools.

During SFT, the model interacted with a stateful BrowserTool that maintained
search result mappings (integer ID → URL) across tool calls within the same
trajectory. The RL tools (search/open/find) are separate classes in verl, but
they share the same instance_id per trajectory. This module provides a shared
state dict keyed by instance_id so that browser.open can resolve integer IDs
from prior browser.search results.

Budget mechanism (two-stage):
  1. SOFT_WARNING_AFTER tool calls: append a warning to every tool response
     telling the model it should wrap up and submit an answer soon.
  2. MAX_TOOL_CALLS: hard cutoff — refuse tool calls entirely and instruct
     the model to answer immediately.
"""

from collections import defaultdict
from typing import Any

# After this many tool calls, a warning is appended to every tool response.
SOFT_WARNING_AFTER = 200

# After this many tool calls, all browser tools refuse and force an answer.
MAX_TOOL_CALLS = 300

BUDGET_WARNING_MSG = (
    "\n\n[SYSTEM] RESEARCH BUDGET WARNING: You have used most of your allowed tool calls. "
    "Start wrapping up NOW. You must submit your final answer soon. "
    "Use the submit_answer tool or write 'Exact Answer:' followed by your answer."
)

BUDGET_EXHAUSTED_MSG = (
    "[SYSTEM] BUDGET EXHAUSTED — NO MORE TOOL CALLS ALLOWED.\n"
    "You MUST submit your answer NOW. Do NOT attempt any more tool calls.\n"
    "Based on everything you have gathered, write your final answer immediately.\n\n"
    "ANY further tool calls will be rejected. ANSWER NOW."
)

# Global per-instance state, shared across all tool instances in the same worker
_sessions: dict[str, dict[str, Any]] = defaultdict(lambda: {
    "search_results": {},  # {int_id: url} from most recent search
    "all_results": {},     # cumulative {int_id: url} across all searches
    "total_tool_calls": 0,
})


def get_session(instance_id: str) -> dict[str, Any]:
    return _sessions[instance_id]


def increment_tool_calls(instance_id: str) -> int:
    """Increment and return total tool call count for this trajectory."""
    session = _sessions[instance_id]
    session["total_tool_calls"] += 1
    return session["total_tool_calls"]


def is_budget_exhausted(instance_id: str) -> bool:
    """Check if this trajectory has exhausted its tool call budget."""
    return _sessions[instance_id]["total_tool_calls"] >= MAX_TOOL_CALLS


def should_warn(instance_id: str) -> bool:
    """Check if this trajectory should get budget warnings appended."""
    count = _sessions[instance_id]["total_tool_calls"]
    return SOFT_WARNING_AFTER <= count < MAX_TOOL_CALLS


def clear_session(instance_id: str) -> None:
    _sessions.pop(instance_id, None)

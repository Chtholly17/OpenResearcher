"""OpenResearcher interaction handler for verl multi-turn RL.

This interaction class manages the dialogue between the RL agent and the
environment. It evaluates the agent's final answer against ground truth
and provides feedback to guide multi-turn exploration.
"""

import re
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction


def extract_answer(content: str) -> Optional[str]:
    """Extract the agent's final answer from its response.

    Supports multiple answer formats used by OpenResearcher:
    - <answer>...</answer> tags
    - "Exact Answer: ..." followed by "Confidence: ..."
    - "Final Answer: ..."
    """
    if not content:
        return None

    # Try <answer> tags first
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try "Exact Answer:" format
    match = re.search(r"Exact Answer:\s*(.*?)(?:\n|Confidence:|$)", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try "Final Answer:" format
    match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def strip_boxed(text: str) -> str:
    """Strip \\boxed{...} wrapper from ground truth answers."""
    match = re.match(r"^\\boxed\{(.*)\}$", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = strip_boxed(text)
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class OpenResearcherInteraction(BaseInteraction):
    """Interaction handler that evaluates research trajectories."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth or kwargs.get("ground_truth", ""),
            "query": kwargs.get("query", ""),
            "best_reward": 0.0,
            "turn_count": 0,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Evaluate the latest assistant message and provide feedback.

        Returns:
            (should_terminate, response_text, reward_score, metadata)
        """
        self._instance_dict[instance_id]["turn_count"] += 1
        ground_truth = self._instance_dict[instance_id]["ground_truth"]

        # Find the latest assistant message
        content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        # Check if the agent has provided a final answer
        answer = extract_answer(content)

        if answer is not None:
            # Agent submitted an answer - evaluate it
            pred_norm = normalize_answer(answer)
            gt_norm = normalize_answer(ground_truth)

            is_correct = (
                pred_norm == gt_norm
                or gt_norm in pred_norm
                or pred_norm in gt_norm
            )

            reward = 1.0 if is_correct else 0.0
            self._instance_dict[instance_id]["best_reward"] = max(
                self._instance_dict[instance_id]["best_reward"], reward
            )

            if is_correct:
                return True, "Your answer is correct!", reward, {"correct": True}
            else:
                # Don't terminate on wrong answer - let the agent try again
                # (up to max_user_turns)
                return False, (
                    "Your answer appears to be incorrect. "
                    "Consider searching for additional sources to verify your findings."
                ), reward, {"correct": False}

        # No answer found in the message - agent is still researching
        # Don't penalize, just acknowledge
        return False, "", 0.0, {"answer_found": False}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Return the best reward achieved during the interaction."""
        return self._instance_dict[instance_id]["best_reward"]

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

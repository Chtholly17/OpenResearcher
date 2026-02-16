"""Research reward tool for verl multi-turn RL.

This is a meta-tool that the agent calls to submit its final answer.
It computes a reward by comparing the submitted answer against ground truth.

The reward function uses fuzzy string matching since research answers are
often free-form text (not exact numbers like in math tasks).
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_reward(prediction: str, ground_truth: str) -> float:
    """Compute reward for a research answer.

    Uses normalized exact match as the primary signal.
    Returns 1.0 for correct, 0.0 for incorrect.

    For more nuanced evaluation (partial credit, semantic similarity),
    this can be extended with an LLM judge or embedding similarity,
    but for GRPO training, binary rewards work well.
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return 0.0

    # Exact match after normalization
    if pred_norm == gt_norm:
        return 1.0

    # Check if ground truth is contained in prediction (common for short answers)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return 1.0

    return 0.0


class ResearchRewardTool(BaseTool):
    """Tool for the agent to submit a final answer and receive reward feedback."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if ground_truth is None:
            ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", "")
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "best_reward": 0.0,
            "submissions": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        ground_truth = self._instance_dict[instance_id]["ground_truth"]
        reward = compute_reward(answer, ground_truth)

        # Track submissions
        self._instance_dict[instance_id]["submissions"].append(answer)

        # Penalize repeated submissions that don't improve
        prev_best = self._instance_dict[instance_id]["best_reward"]
        tool_reward = 0.0
        if reward > prev_best:
            tool_reward = reward - prev_best  # Positive signal for improvement
            self._instance_dict[instance_id]["best_reward"] = reward
        elif len(self._instance_dict[instance_id]["submissions"]) > 1:
            tool_reward = -0.05  # Small penalty for non-improving resubmission

        feedback = f"Answer submitted. Current reward: {reward:.1f}"
        if reward == 1.0:
            feedback = "Correct! Your answer matches the ground truth."
        elif reward == 0.0 and len(self._instance_dict[instance_id]["submissions"]) > 1:
            feedback = "Incorrect. Consider searching for more information and refining your answer."

        return ToolResponse(text=feedback), tool_reward, {"reward": reward}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["best_reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

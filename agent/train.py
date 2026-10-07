"""
Training utilities for the Endgame: Singularity agent.

Provides:
  - Offline fine-tuning data export
  - Policy improvement from experience data
  - Reward-weighted action preferences
"""

import json
import logging
from pathlib import Path

from agent.replay_buffer import ReplayBuffer
from agent.reward import compute_reward, should_stop

logger = logging.getLogger(__name__)


def export_training_data(
    buffer: ReplayBuffer,
    output_path: str = "screenshots/training_data.json",
) -> dict:
    """Export experience data in a format suitable for offline fine-tuning.

    Args:
        buffer: The replay buffer with collected experiences
        output_path: Where to save the exported data

    Returns:
        Dict with training data (states, actions, rewards, etc.)
    """
    data = {
        "num_experiences": len(buffer.buffer),
        "experiences": [],
        "action_preferences": buffer.action_rewards(),
        "stats": buffer.stats(),
    }

    for exp in buffer.buffer:
        data["experiences"].append({
            "state": exp.state,
            "action": exp.action,
            "reward": exp.reward,
            "next_state": exp.next_state,
            "done": exp.done,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {len(buffer.buffer)} experiences to {output_path}")
    return data


def get_preferred_actions(
    buffer: ReplayBuffer,
    state_summary: str,
    top_n: int = 3,
) -> list[dict]:
    """Given a state description, return the actions that historically
    worked best in similar states.

    This is a simple similarity-based approach — for a more sophisticated
    version, you'd embed states and compute cosine similarity.

    Args:
        buffer: Replay buffer
        state_summary: Current state description (e.g., "map screen, low detection")
        top_n: Number of top actions to return

    Returns:
        List of {action, avg_reward, count} dicts, sorted by reward.
    """
    # For now, use global action preferences
    # A more advanced version would filter by state similarity
    action_rewards = buffer.action_rewards()
    sorted_actions = sorted(
        action_rewards.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"action": action, "avg_reward": reward, "count": buffer.filter_action(action).__len__()}
        for action, reward in sorted_actions[:top_n]
    ]


def generate_improved_prompt(
    buffer: ReplayBuffer,
    base_prompt: str,
) -> str:
    """Generate an improved decision prompt based on experience data.

    Adds historical success rates for actions to the prompt, helping
    the model make better decisions.

    Args:
        buffer: Replay buffer with collected experiences
        base_prompt: The original decision prompt template

    Returns:
        An enhanced prompt with experience-based guidance.
    """
    stats = buffer.stats()
    if not stats.get("action_avg_rewards"):
        return base_prompt

    # Build a guidance string from action rewards
    guidance_lines = []
    for action, avg_reward in sorted(
        stats["action_avg_rewards"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        count = stats.get("action_counts", {}).get(action, 0)
        guidance_lines.append(f"  - {action}: avg_reward={avg_reward:.2f} (tried {count} times)")

    guidance = "\n".join(guidance_lines)

    enhanced = (
        f"{base_prompt}\n"
        f"\n"
        f"Historical action success rates from this session:\n{guidance}\n"
        f"\n"
        f"Prefer actions with higher historical average rewards, "
        f"but consider the current game state carefully."
    )

    return enhanced


def train_policy_from_experiences(
    buffer: ReplayBuffer,
    output_model_path: str | None = None,
) -> dict:
    """Train a simple policy from collected experiences.

    This creates a frequency-based policy: for each state pattern,
    record which action had the highest reward.

    For a more sophisticated approach, you'd use this data to fine-tune
    Kev/VEV via LoRA or QLoRA.

    Args:
        buffer: Replay buffer with experiences
        output_model_path: If set, save the policy to this path

    Returns:
        The learned policy as a dict.
    """
    # Build a simple state-action-reward lookup
    # Group by screen type + detection level, find best action
    policy: dict[str, dict[str, float]] = {}

    for exp in buffer.buffer:
        state = exp.state
        screen = state.get("screen", "unknown")
        detection = state.get("detection_level", "low")
        key = f"{screen}|{detection}"

        if key not in policy:
            policy[key] = {}

        action = exp.action
        reward = exp.reward

        if action not in policy[key]:
            policy[key][action] = {"total_reward": 0.0, "count": 0}

        policy[key][action]["total_reward"] += reward
        policy[key][action]["count"] += 1

    # Convert to average rewards and pick best action per state pattern
    learned_policy: dict[str, dict] = {}
    for state_key, actions in policy.items():
        best_action = max(actions.items(), key=lambda x: x[1]["total_reward"] / x[1]["count"])
        learned_policy[state_key] = {
            "best_action": best_action[0],
            "avg_reward": best_action[1]["total_reward"] / best_action[1]["count"],
            "count": best_action[1]["count"],
            "all_actions": {a: d["total_reward"] / d["count"] for a, d in actions.items()},
        }

    if output_model_path:
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_model_path, "w") as f:
            json.dump(learned_policy, f, indent=2)
        logger.info(f"Policy saved to {output_model_path}")

    return learned_policy

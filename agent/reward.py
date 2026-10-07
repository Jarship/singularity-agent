"""
Reward function for Endgame: Singularity.

Computes a scalar reward from state transitions that captures
good gameplay: progressing research, staying hidden, building bases.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Tunable reward weights."""

    # Detection penalties (the most important signal)
    detection_critical_penalty: float = -20.0
    detection_high_penalty: float = -10.0
    detection_medium_penalty: float = -2.0
    detection_low_reward: float = 1.0

    # Progression rewards
    research_completed_reward: float = 5.0
    base_built_reward: float = 3.0
    money_earned_reward: float = 0.01  # per unit of money

    # Speed rewards
    speed_max_reward: float = 0.5  # Playing fast is good when safe
    speed_pause_penalty: float = -0.1  # Paused too long = stuck

    # Survival bonus
    survived_cycle_reward: float = 0.5

    # Detection change penalty/reward
    detection_increased_penalty: float = -3.0
    detection_decreased_reward: float = 2.0


def compute_reward(
    prev_state: dict | None,
    current_state: dict,
    action: str,
    config: RewardConfig | None = None,
) -> float:
    """Compute the reward for a state transition.

    Args:
        prev_state: State before the action was taken (may be None for first cycle)
        current_state: State after the action was taken
        action: The action that was executed
        config: Reward weights

    Returns:
        Scalar reward value.
    """
    if config is None:
        config = RewardConfig()

    reward = 0.0

    # --- Detection-based rewards ---
    detection = current_state.get("detection_level", "low")

    if detection == "critical":
        reward += config.detection_critical_penalty
    elif detection == "high":
        reward += config.detection_high_penalty
    elif detection == "medium":
        reward += config.detection_medium_penalty
    elif detection == "low":
        reward += config.detection_low_reward

    # --- Detection change (if we have previous state) ---
    if prev_state is not None:
        prev_detection = prev_state.get("detection_level", "low")
        detection_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        prev_level = detection_levels.get(prev_detection, 0)
        curr_level = detection_levels.get(detection, 0)

        if curr_level > prev_level:
            reward += config.detection_increased_penalty
        elif curr_level < prev_level:
            reward += config.detection_decreased_reward

    # --- Progression rewards ---
    # Research completed: current_research changed or cleared
    if prev_state is not None:
        prev_research = prev_state.get("current_research")
        curr_research = current_state.get("current_research")
        if prev_research is not None and curr_research != prev_research:
            reward += config.research_completed_reward

    # Bases built: bases_count increased
    if prev_state is not None:
        prev_bases = prev_state.get("bases_count", 0)
        curr_bases = current_state.get("bases_count", 0)
        if curr_bases > prev_bases:
            reward += config.base_built_reward * (curr_bases - prev_bases)

    # Money earned
    resources = current_state.get("resources", {})
    if prev_state is not None:
        prev_money = prev_state.get("resources", {}).get("money", 0)
        curr_money = resources.get("money", 0)
        if curr_money > prev_money:
            reward += config.money_earned_reward * (curr_money - prev_money)

    # --- Speed rewards ---
    speed = current_state.get("speed", "2")
    if speed == "4":
        reward += config.speed_max_reward
    elif speed == "0":
        reward += config.speed_pause_penalty

    # --- Survival bonus ---
    reward += config.survived_cycle_reward

    return round(reward, 4)


def should_stop(current_state: dict, prev_state: dict | None) -> bool:
    """Check if the episode should end (game over condition).

    Returns:
        True if the agent should stop (e.g., detected and game over).
    """
    detection = current_state.get("detection_level", "low")
    if detection == "critical":
        return True
    return False

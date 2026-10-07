"""
Action mapping for Endgame: Singularity.

Maps high-level agent decisions to key sequences that the game understands.
"""

# High-level actions → key sequences
# These are the actions the agent can take.
# Each maps to a list of keys to press in sequence.

ACTION_MAP: dict[str, dict] = {
    # Speed control
    "speed_max": {
        "keys": ["4"],
        "description": "Set game speed to maximum (4)",
        "pause_after": False,
    },
    "speed_pause": {
        "keys": ["0"],
        "description": "Pause the game (speed 0)",
        "pause_after": False,
    },
    "speed_slow": {
        "keys": ["1"],
        "description": "Set game speed to slow (1)",
        "pause_after": False,
    },
    "speed_toggle": {
        "keys": ["Space"],
        "description": "Toggle pause / unpause",
        "pause_after": True,
    },

    # Navigation / confirmation
    "confirm": {
        "keys": ["Return"],
        "description": "Confirm current choice (Enter key)",
        "pause_after": True,
    },
    "cancel": {
        "keys": ["Escape"],
        "description": "Cancel / leave current dialog",
        "pause_after": False,
    },
    "cancel_twice": {
        "keys": ["Escape", "Escape"],
        "description": "Cancel twice (back to map screen)",
        "pause_after": True,
    },

    # Research
    "research_next": {
        "keys": ["Tab"],
        "description": "Navigate to next research option in the list",
        "pause_after": False,
    },
    "research_confirm": {
        "keys": ["Return"],
        "description": "Confirm research selection",
        "pause_after": True,
    },
    "research_abort": {
        "keys": ["Escape"],
        "description": "Abort current research",
        "pause_after": False,
    },

    # Base management
    "build_base": {
        "keys": ["B"],
        "description": "Open build menu for base construction",
        "pause_after": False,
    },
    "build_confirm": {
        "keys": ["Return"],
        "description": "Confirm base construction",
        "pause_after": True,
    },

    # Emergency
    "abort_mission": {
        "keys": ["Escape", "Escape"],
        "description": "Abort current mission / return to map",
        "pause_after": True,
    },
    "emergency_pause": {
        "keys": ["0"],
        "description": "Emergency pause (detection critical)",
        "pause_after": False,
    },
}

# Which actions are "safe" when detection is high
SAFE_ACTIONS_UNDER_DETECTION = {
    "speed_pause",
    "cancel",
    "cancel_twice",
    "abort_mission",
    "emergency_pause",
}


def get_action_keys(action_name: str) -> list[str]:
    """Get the key sequence for a high-level action."""
    action = ACTION_MAP.get(action_name)
    if action is None:
        raise ValueError(f"Unknown action: {action_name}. Available: {list(ACTION_MAP.keys())}")
    return action["keys"]


def get_action_description(action_name: str) -> str:
    """Get the human-readable description of an action."""
    action = ACTION_MAP.get(action_name)
    if action is None:
        return f"Unknown action: {action_name}"
    return action["description"]


def should_pause_after(action_name: str) -> bool:
    """Should the agent wait after executing this action?"""
    action = ACTION_MAP.get(action_name)
    if action is None:
        return True
    return action.get("pause_after", False)


def is_safe_under_detection(action_name: str) -> bool:
    """Is this action safe when detection level is high?"""
    return action_name in SAFE_ACTIONS_UNDER_DETECTION


def get_all_actions() -> list[str]:
    """Return the list of all available high-level actions."""
    return list(ACTION_MAP.keys())

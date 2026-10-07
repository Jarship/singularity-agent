"""
Key injection for Endgame: Singularity.

Sends keyboard input to the game window using xdotool (Linux/X11).
Falls back to pyautogui if xdotool is unavailable.
"""

import subprocess
import time
from dataclasses import dataclass


@dataclass
class InputConfig:
    """Configuration for key injection."""

    key_press_delay: float = 0.3  # seconds between key presses
    window_title: str = "Endgame: Singularity"


def _find_window_id(window_title: str) -> str | None:
    """Find the X11 window ID for the game window."""
    result = subprocess.run(
        ["xdotool", "search", "--name", window_title],
        capture_output=True,
        text=True,
    )
    window_ids = result.stdout.strip().split("\n")
    return window_ids[0] if window_ids and window_ids[0] else None


def send_keys(
    keys: list[str],
    window_title: str = "Endgame: Singularity",
    delay: float = 0.3,
) -> bool:
    """Send a sequence of key presses to the game window.

    Args:
        keys: list of key names (xdotool syntax, e.g. "Return", "Escape", "a", "4")
        window_title: partial title of the game window
        delay: seconds between each key press

    Returns:
        True if all keys were sent successfully, False otherwise.
    """
    win_id = _find_window_id(window_title)
    if not win_id:
        # Fallback: try pyautogui
        return _send_keys_pyautogui(keys, delay)

    for key in keys:
        # Activate the window first (bring to front without stealing focus permanently)
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", win_id],
            capture_output=True,
        )
        time.sleep(0.05)

        # Send the key
        result = subprocess.run(
            ["xdotool", "key", "--window", win_id, key],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Fallback to pyautogui for this key
            _send_keys_pyautogui([key], delay)

        time.sleep(delay)

    return True


def _send_keys_pyautogui(keys: list[str], delay: float = 0.3) -> bool:
    """Fallback key injection using pyautogui."""
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        for key in keys:
            pyautogui.press(key)
            time.sleep(delay)
        return True
    except ImportError:
        print(f"[ERROR] Cannot send key '{key}': neither xdotool nor pyautogui available")
        return False


def press_key(
    key: str,
    window_title: str = "Endgame: Singularity",
    delay: float = 0.3,
) -> bool:
    """Send a single key press."""
    return send_keys([key], window_title=window_title, delay=delay)


def hold_and_release(
    key: str,
    hold_time: float = 0.5,
    window_title: str = "Endgame: Singularity",
) -> bool:
    """Hold a key for a duration, then release it."""
    win_id = _find_window_id(window_title)
    if not win_id:
        import pyautogui

        pyautogui.keyDown(key)
        time.sleep(hold_time)
        pyautogui.keyUp(key)
        return True

    subprocess.run(
        ["xdotool", "windowactivate", "--sync", win_id],
        capture_output=True,
    )
    time.sleep(0.05)

    subprocess.run(
        ["xdotool", "keydown", "--window", win_id, key],
        capture_output=True,
    )
    time.sleep(hold_time)
    subprocess.run(
        ["xdotool", "keyup", "--window", win_id, key],
        capture_output=True,
    )
    return True

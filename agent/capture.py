"""
Screenshot capture for Endgame: Singularity.

Uses mss for fast screen capture and PIL for image handling.
Targets the game window by name via X11.
"""

import time
from pathlib import Path

import mss
import numpy as np
from PIL import Image


def capture_screen(
    region: dict | None = None,
    output_path: str | None = None,
) -> Image.Image:
    """Capture a screenshot of the screen (or a region).

    Args:
        region: dict with left, top, width, height. None = full screen.
        output_path: if set, save the screenshot to this path.

    Returns:
        PIL Image of the captured area.
    """
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]  # primary monitor
        else:
            monitor = region
        screenshot = sct.grab(monitor)

    # Convert to PIL Image
    img = Image.frombytes(
        "RGB",
        (screenshot.width, screenshot.height),
        screenshot.bgra,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    return img


def capture_game_window(
    window_title: str = "Endgame: Singularity",
    output_path: str | None = None,
) -> Image.Image | None:
    """Capture the game window by searching for it on screen.

    Uses xdotool to find the window geometry, then captures that region.

    Returns:
        PIL Image of the game window, or None if window not found.
    """
    import subprocess

    # Find window geometry via xdotool
    result = subprocess.run(
        ["xdotool", "search", "--name", window_title],
        capture_output=True,
        text=True,
    )

    window_ids = result.stdout.strip().split("\n")
    if not window_ids or not window_ids[0]:
        return None

    # Get window geometry for the first match
    win_id = window_ids[0]
    geo_result = subprocess.run(
        ["xdotool", "getwindowgeometry", win_id],
        capture_output=True,
        text=True,
    )

    # Parse geometry output like "  Position: 100,200  Geometry: 1280x720"
    geo_text = geo_result.stdout.strip()
    geometry_line = [l for l in geo_text.split("\n") if "Geometry:" in l]
    if not geometry_line:
        return None

    # Extract WxH and X,Y
    import re

    geom_match = re.search(r"Geometry:\s*(\d+)x(\d+)\s*\+\d+\+\d+", geometry_line[0])
    if not geom_match:
        return None

    width = int(geom_match.group(1))
    height = int(geom_match.group(2))

    # Get position
    pos_match = re.search(r"Position:\s*(\d+),(\d+)", geo_text)
    if not pos_match:
        return None

    x = int(pos_match.group(1))
    y = int(pos_match.group(2))

    region = {
        "left": x,
        "top": y,
        "width": width,
        "height": height,
    }

    return capture_screen(region=region, output_path=output_path)


def capture_full_screen(output_path: str | None = None) -> Image.Image:
    """Capture the entire primary monitor."""
    return capture_screen(region=None, output_path=output_path)

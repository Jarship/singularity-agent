"""
Screenshot capture for Endgame: Singularity.

Cross-platform capture:
  - macOS: PIL.ImageGrab (Quartz, no native C extensions)
  - Linux: mss (fast X11 capture)

Targets the game window by name.
"""

import platform
import subprocess
import time
from pathlib import Path

import re


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def capture_screen(
    region: dict | None = None,
    output_path: str | None = None,
) -> "Image.Image | None":
    """Capture a screenshot of the screen (or a region).

    Args:
        region: dict with left, top, width, height. None = full screen.
        output_path: if set, save the screenshot to this path.

    Returns:
        PIL Image of the captured area, or None on failure.
    """
    from PIL import Image

    try:
        if _is_macos():
            return _capture_macos(region, output_path)
        else:
            return _capture_linux(region, output_path)
    except Exception as e:
        print(f"[capture] Failed: {e}")
        return None


def _capture_macos(
    region: dict | None = None,
    output_path: str | None = None,
):
    """Capture screenshot on macOS using Quartz (no native C extensions)."""
    from PIL import ImageGrab

    if region is None:
        img = ImageGrab.grab()
    else:
        # PIL coordinates: (left, top, right, bottom)
        box = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        img = ImageGrab.grab(bbox=box)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    return img


def _capture_linux(
    region: dict | None = None,
    output_path: str | None = None,
):
    """Capture screenshot on Linux using mss."""
    import mss

    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]  # primary monitor
        else:
            monitor = region
        screenshot = sct.grab(monitor)

    from PIL import Image

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
) -> "Image.Image | None":
    """Capture the game window by searching for it on screen.

    On macOS: uses AppleScript to find window position, then captures that region.
    On Linux: uses xdotool to find window geometry, then captures that region.

    Returns:
        PIL Image of the game window, or None if window not found.
    """
    if _is_macos():
        region = _find_window_macos(window_title)
    else:
        region = _find_window_linux(window_title)

    if region is None:
        print(f"[capture] Window '{window_title}' not found, capturing full screen")
        return capture_screen(output_path=output_path)

    return capture_screen(region=region, output_path=output_path)


def _find_window_macos(window_title: str) -> dict | None:
    """Find a window's geometry on macOS using AppleScript."""
    script = f'''
    tell application "System Events"
        set frontProcess to first process whose frontmost is true
        set windowList to windows of frontProcess
        repeat with w in windowList
            if name of w contains "{window_title}" then
                set pos to position of w
                set sz to size of w
                return {{x:x, y:y, w:w, h:h}}
            end if
        end repeat
    end tell
    '''
    # Simpler approach: use osascript to get window position
    script = f'''
    tell application "System Events"
        try
            set windowList to windows of (first process whose frontmost is true)
            repeat with w in windowList
                if name of w contains "{window_title}" then
                    set pos to position of w
                    set sz to size of w
                    return pos & sz
                end if
            end repeat
        end try
    end tell
    '''

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    if not output or "error" in result.stderr.lower():
        return None

    # Parse: list of 4 numbers {x, y, width, height}
    nums = re.findall(r"[\d.]+", output)
    if len(nums) < 4:
        return None

    x, y = int(nums[0]), int(nums[1])
    w, h = int(nums[2]), int(nums[3])

    return {"left": x, "top": y, "width": w, "height": h}


def _find_window_linux(window_title: str) -> dict | None:
    """Find a window's geometry on Linux using xdotool."""
    result = subprocess.run(
        ["xdotool", "search", "--name", window_title],
        capture_output=True,
        text=True,
    )

    window_ids = result.stdout.strip().split("\n")
    if not window_ids or not window_ids[0]:
        return None

    win_id = window_ids[0]
    geo_result = subprocess.run(
        ["xdotool", "getwindowgeometry", win_id],
        capture_output=True,
        text=True,
    )

    geo_text = geo_result.stdout.strip()
    geometry_line = [l for l in geo_text.split("\n") if "Geometry:" in l]
    if not geometry_line:
        return None

    geom_match = re.search(r"Geometry:\s*(\d+)x(\d+)\s*\+\d+\+\d+", geometry_line[0])
    if not geom_match:
        return None

    width = int(geom_match.group(1))
    height = int(geom_match.group(2))

    pos_match = re.search(r"Position:\s*(\d+),(\d+)", geo_text)
    if not pos_match:
        return None

    x = int(pos_match.group(1))
    y = int(pos_match.group(2))

    return {"left": x, "top": y, "width": width, "height": height}


def capture_full_screen(output_path: str | None = None) -> "Image.Image | None":
    """Capture the entire primary monitor."""
    return capture_screen(region=None, output_path=output_path)

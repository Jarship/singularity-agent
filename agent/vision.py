"""
Ollama integration for the Endgame: Singularity agent.

Provides two functions:
  - describe_state(): send a screenshot to the vision model, get a text description
  - decide_action(): send the description + available actions, get the next action
"""

import json
import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    vision_model: str | None = None
    decision_model: str = "qwen2.5:latest"
    timeout: int = 300


def _ollama_generate(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    config: OllamaConfig | None = None,
) -> str:
    """Send a request to Ollama's generate API.

    Args:
        model: model name (e.g. 'qwen2.5vl:2b')
        prompt: text prompt
        images: list of base64-encoded image strings (optional)
        config: OllamaConfig instance

    Returns:
        The model's text response.
    """
    if config is None:
        config = OllamaConfig()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # low temp for consistent decisions
            "num_predict": 500,
        },
    }

    if images:
        payload["images"] = images

    try:
        resp = requests.post(
            f"{config.host}/api/generate",
            json=payload,
            timeout=config.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {config.host}. "
            f"Is Ollama running? Try: curl {config.host}/api/tags"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama request timed out after {config.timeout}s")


def _analyze_screenshot_pixels(screenshot_path: str) -> dict:
    """Analyze a screenshot's pixels to extract basic game state.

    Uses color and position heuristics to detect:
    - Detection level (red warning pixels in center regions)
    - Speed indicator (bright UI elements)
    - Resource presence (gold/copper colors)
    - Screen type (map vs base vs research vs dialog)

    Returns a state dict suitable for the text-only decision model.
    """
    from PIL import Image
    import numpy as np

    img = Image.open(screenshot_path)
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    h, w = arr.shape[:2]

    # --- Detection level: look for red pixels in center region ---
    center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    red_mask = (center[:, :, 0] > 180) & (center[:, :, 1] < 80) & (center[:, :, 2] < 80)
    red_ratio = red_mask.sum() / center.shape[0] / center.shape[1]

    if red_ratio > 0.05:
        detection_level = "critical"
    elif red_ratio > 0.02:
        detection_level = "high"
    elif red_ratio > 0.005:
        detection_level = "medium"
    else:
        detection_level = "low"

    # --- Speed: look for bright pixels in bottom-center (speed indicator) ---
    bottom_center = arr[3 * h // 4 :, w // 4 : 3 * w // 4]
    bright_mask = (
        (bottom_center[:, :, 0] > 200)
        & (bottom_center[:, :, 1] > 200)
        & (bottom_center[:, :, 2] > 200)
    )
    bright_ratio = bright_mask.sum() / bottom_center.shape[0] / bottom_center.shape[1]

    if bright_ratio > 0.1:
        speed = "4"
    elif bright_ratio > 0.05:
        speed = "3"
    elif bright_ratio > 0.02:
        speed = "2"
    elif bright_ratio > 0.005:
        speed = "1"
    else:
        speed = "0"

    # --- Resources: look for gold/copper in top-right ---
    top_right = arr[: h // 4, 3 * w // 4 :]
    gold_mask = (
        (top_right[:, :, 0] > 180)
        & (top_right[:, :, 1] > 140)
        & (top_right[:, :, 2] < 100)
    )
    has_resources = gold_mask.sum() > 100

    # --- Screen type: detect dialog by looking for dark overlay ---
    dark_pixels = (arr.mean(axis=2) < 50).sum()
    total_pixels = arr.shape[0] * arr.shape[1]
    dark_ratio = dark_pixels / total_pixels

    if dark_ratio > 0.3:
        screen = "dialog"
    elif has_resources and detection_level == "low":
        screen = "base"
    elif detection_level in ("medium", "high", "critical"):
        screen = "research"
    else:
        screen = "map"

    # --- Available actions (basic set based on screen type) ---
    base_actions = ["speed_max", "speed_pause", "speed_slow", "confirm", "cancel"]
    if screen == "base":
        base_actions += ["build_base", "research_next"]
    elif screen == "research":
        base_actions += ["research_next", "research_confirm"]

    return {
        "screen": screen,
        "speed": speed,
        "detection_level": detection_level,
        "resources": {
            "money": 100 if has_resources else 0,
            "cpu": 50 if has_resources else 0,
            "research_points": 25 if has_resources else 0,
        },
        "available_actions": base_actions,
        "bases_count": 1 if screen == "base" else 0,
        "current_research": None,
        "detection_warning": "red_alert" if detection_level in ("critical", "high") else None,
    }


def describe_state(
    screenshot_path: str,
    config: OllamaConfig | None = None,
) -> tuple[dict, str]:
    """Send a game screenshot to the vision model and get a structured state description.

    When vision_model is null, returns a minimal default state without
    sending the image to Ollama (text-only mode).

    Args:
        screenshot_path: path to the screenshot PNG file
        config: OllamaConfig instance

    Returns:
        Tuple of (parsed state dict, raw model reasoning text).
    """
    if config is None:
        config = OllamaConfig()

    # Text-only mode: analyze screenshot pixels for basic state, then decide
    if config.vision_model is None:
        logger.info("Vision model disabled — using pixel analysis + text-only mode")
        state = _analyze_screenshot_pixels(screenshot_path)
        return state, "text-only mode: pixel analysis (no vision model)"

    # Read image, resize for Ollama (full screenshots are too large)
    import base64
    from PIL import Image

    img = Image.open(screenshot_path)
    max_width = 480
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Encode as JPEG (smaller than PNG)
    import io
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are analyzing a screenshot of the game Endgame: Singularity.\n"
        "Describe the current game state in exactly these fields as JSON:\n"
        '{"screen": "map|base|research|dialog", "speed": "0|1|2|3|4", '
        '"detection_level": "low|medium|high|critical", '
        '"resources": {"money": <int>, "cpu": <int>, "research_points": <int>}, '
        '"available_actions": ["list", "of", "visible", "actions"], '
        '"bases_count": <int>, "current_research": "<name or null>", '
        '"detection_warning": "<text or null>"}\n'
        "Be concise. Only output valid JSON. Do not include markdown code blocks."
    )

    response = _ollama_generate(
        model=config.vision_model,
        prompt=prompt,
        images=[image_b64],
        config=config,
    )

    # Try to parse JSON from the response
    try:
        # Strip markdown code blocks if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            if clean.startswith("json"):
                clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0].strip()
        state = json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        # Return raw response if JSON parsing fails
        state = {"raw_response": response, "error": "json_parse_failed"}

    return state, response


def decide_action(
    state_description: dict,
    available_actions: list[str],
    config: OllamaConfig | None = None,
) -> tuple[str, str]:
    """Given a game state description, decide the next action.

    Args:
        state_description: dict from describe_state()
        available_actions: list of high-level actions the agent can take
        config: OllamaConfig instance

    Returns:
        Tuple of (chosen action name, raw model reasoning text).
    """
    if config is None:
        config = OllamaConfig()

    detection = state_description.get("detection_level", "low")
    speed = state_description.get("speed", "2")
    actions_str = ", ".join(available_actions)

    prompt = (
        f"Endgame: Singularity game state:\n"
        f"  Screen: {state_description.get('screen', 'unknown')}\n"
        f"  Speed: {speed}\n"
        f"  Detection: {detection}\n"
        f"  Bases: {state_description.get('bases_count', 0)}\n"
        f"  Current research: {state_description.get('current_research', 'none')}\n"
        f"  Detection warning: {state_description.get('detection_warning', 'none')}\n"
        f"  Resources: {state_description.get('resources', {})}\n"
        f"\n"
        f"Available actions: {actions_str}\n"
        f"\n"
        f"Rules:\n"
        f"  - If detection is critical or high, choose an action that reduces detection (abort_mission, speed_pause)\n"
        f"  - If speed is not max and no urgent threat, choose speed_max\n"
        f"  - Otherwise, choose the best research/build action from the available list\n"
        f"  - Only output the action name. No explanation.\n"
        f"\n"
        f"Chosen action:"
    )

    response = _ollama_generate(
        model=config.decision_model,
        prompt=prompt,
        config=config,
    )

    return response.strip().split("\n")[0].strip(), response

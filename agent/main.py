"""
Main agent loop for Endgame: Singularity.

Orchestrates the pipeline:
  capture → describe → decide → execute → wait
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

from agent.capture import capture_full_screen, capture_game_window
from agent.input import send_keys
from agent.vision import OllamaConfig, describe_state, decide_action
from agent.action_map import (
    get_action_keys,
    get_action_description,
    should_pause_after,
    get_all_actions,
    SAFE_ACTIONS_UNDER_DETECTION,
)
from agent.replay_buffer import ReplayBuffer, Experience
from agent.reward import compute_reward, should_stop
from agent.train import (
    export_training_data,
    generate_improved_prompt,
    get_preferred_actions,
    train_policy_from_experiences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Global shutdown flag for signal handling
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown signal received, stopping after current cycle...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_screenshot_path(cycle: int) -> str:
    """Generate a path for saving a screenshot."""
    return f"screenshots/cycle_{cycle:06d}.png"


def run_agent_loop(config: dict, max_cycles: int | None = None):
    """Run the main agent loop.

    Args:
        config: loaded configuration dict
        max_cycles: if set, stop after this many decision cycles
    """
    # Extract config values with defaults
    game_window = config.get("display", {}).get("game_window_title", "Endgame: Singularity")
    decision_interval = config.get("timing", {}).get("decision_interval", 2.0)
    key_press_delay = config.get("timing", {}).get("key_press_delay", 0.3)
    max_retries = config.get("timing", {}).get("max_retries", 3)
    screenshot_interval = config.get("timing", {}).get("screenshot_interval", 5.0)
    log_decisions = config.get("logging", {}).get("log_decisions", True)
    screenshots_dir = config.get("logging", {}).get("screenshots_dir", "screenshots")

    # Ollama config
    ollama_cfg_data = config.get("ollama", {})
    ollama_config = OllamaConfig(
        host=ollama_cfg_data.get("host", "http://localhost:11434"),
        vision_model=ollama_cfg_data.get("vision_model", "qwen2.5vl:2b"),
        decision_model=ollama_cfg_data.get("decision_model", "qwen2.5vl:2b"),
    )

    # Ensure screenshots directory exists
    Path(screenshots_dir).mkdir(parents=True, exist_ok=True)

    # Training: initialize replay buffer
    training_enabled = config.get("training", {}).get("enabled", True)
    replay_buffer = ReplayBuffer(
        max_size=config.get("training", {}).get("buffer_size", 10_000),
        save_path=f"{screenshots_dir}/experiences.jsonl",
    )
    reward_cfg_data = config.get("reward", {})
    from agent.reward import RewardConfig
    reward_config = RewardConfig(**reward_cfg_data) if reward_cfg_data else RewardConfig()

    # Training: load existing policy if available
    policy_path = config.get("training", {}).get("policy_path", None)
    learned_policy: dict | None = None
    if policy_path and Path(policy_path).exists():
        with open(policy_path) as f:
            learned_policy = json.load(f)
        logger.info(f"Loaded learned policy from {policy_path} ({len(learned_policy)} state patterns)")

    # Use improved prompt if we have experience data
    decision_prompt_override = None
    if training_enabled and len(replay_buffer.buffer) > 10:
        base_prompt = config.get("decision", {}).get("decision_prompt", "")
        if base_prompt:
            decision_prompt_override = generate_improved_prompt(replay_buffer, base_prompt)

    # Available actions the agent can choose from
    available_actions = get_all_actions()

    logger.info("=" * 60)
    logger.info("Endgame: Singularity — AI Agent Starting")
    logger.info(f"Game window: {game_window}")
    logger.info(f"Ollama host: {ollama_config.host}")
    logger.info(f"Vision model: {ollama_config.vision_model}")
    logger.info(f"Available actions: {len(available_actions)}")
    if training_enabled:
        logger.info(f"Training enabled: {len(replay_buffer.buffer)} prior experiences")
        if learned_policy:
            logger.info(f"Learned policy loaded: {len(learned_policy)} state patterns")
    logger.info("=" * 60)

    cycle = 0
    last_screenshot_time = 0.0
    last_description: dict | None = None
    prev_state: dict | None = None
    episode_reward = 0.0

    try:
        while True:
            if max_cycles is not None and cycle >= max_cycles:
                logger.info(f"Reached max cycles ({max_cycles}). Stopping.")
                break

            if _shutdown_requested:
                logger.info("Shutdown requested. Stopping.")
                break

            cycle += 1
            logger.info(f"--- Cycle {cycle} ---")

            # Step 1: Capture screenshot
            screenshot_path = get_screenshot_path(cycle)
            logger.info("Capturing screenshot...")
            img = capture_game_window(
                window_title=game_window,
                output_path=screenshot_path,
            )

            if img is None:
                logger.error("Game window not found! Is Endgame: Singularity running?")
                logger.info("Trying full-screen capture instead...")
                img = capture_full_screen(output_path=screenshot_path)

            if img is None:
                logger.error("Failed to capture any screenshot. Retrying in 5s...")
                time.sleep(5)
                continue

            logger.info(f"Screenshot saved: {screenshot_path} ({img.width}x{img.height})")

            # Step 2: Describe game state via Ollama vision model
            logger.info("Describing game state...")
            try:
                state, state_reasoning = describe_state(screenshot_path, config=ollama_config)
                last_description = state
                logger.info(f"State: {json.dumps(state, indent=2)}")
                logger.info(f"Vision reasoning:\n{state_reasoning}")
            except Exception as e:
                logger.error(f"Failed to describe state: {e}")
                time.sleep(decision_interval)
                continue

            # Step 3: Decide action
            logger.info("Deciding next action...")
            action = None

            # Check if we should stop (detection critical)
            if prev_state and should_stop(state, prev_state):
                logger.warning("Detection critical! Episode ending.")
                action = "emergency_pause"
                keys = get_action_keys(action)
                logger.info(f"Emergency: {action} → keys={keys}")
                send_keys(keys, window_title=game_window, delay=key_press_delay)
                exp = Experience(
                    cycle=cycle,
                    timestamp=time.time(),
                    state=prev_state,
                    action="emergency_pause",
                    keys_sent=keys,
                    reward=-20.0,
                    next_state=state,
                    done=True,
                )
                if training_enabled:
                    replay_buffer.append(exp)
                    episode_reward += exp.reward
                logger.info(f"Episode ended. Total reward: {episode_reward:.2f}")
                break

            # Use learned policy if available
            if learned_policy:
                screen = state.get("screen", "unknown")
                detection = state.get("detection_level", "low")
                policy_key = f"{screen}|{detection}"
                if policy_key in learned_policy:
                    policy_entry = learned_policy[policy_key]
                    best = policy_entry["best_action"]
                    logger.info(
                        f"Policy suggests '{best}' "
                        f"(avg_reward={policy_entry['avg_reward']:.2f}, "
                        f"tried {policy_entry['count']} times)"
                    )
                    if detection in ("low", "medium") or best in SAFE_ACTIONS_UNDER_DETECTION:
                        action = best

            # If policy didn't pick an action, use the LLM
            if action is None:
                for attempt in range(max_retries):
                    try:
                        action, decision_reasoning = decide_action(
                            state, available_actions, config=ollama_config
                        )
                        logger.info(f"Chosen action: {action}")
                        logger.info(f"Decision reasoning:\n{decision_reasoning}")
                        break
                    except Exception as e:
                        logger.warning(f"Decision attempt {attempt + 1} failed: {e}")
                        if attempt == max_retries - 1:
                            logger.error("All decision attempts failed. Skipping cycle.")
                            action = None

            if action is None:
                time.sleep(decision_interval)
                continue

            # Step 4: Validate action
            if action not in available_actions:
                logger.warning(
                    f"Model returned unknown action '{action}'. "
                    f"Falling back to 'speed_pause'."
                )
                action = "speed_pause"

            # Step 5: Execute action
            keys = get_action_keys(action)
            logger.info(
                f"Executing: {action} → keys={keys} "
                f"({get_action_description(action)})"
            )

            if log_decisions:
                with open(f"{screenshots_dir}/decision_log.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "cycle": cycle,
                                "action": action,
                                "keys": keys,
                                "state": state,
                                "state_reasoning": state_reasoning,
                                "decision_reasoning": decision_reasoning,
                                "timestamp": time.time(),
                            }
                        )
                        + "\n"
                    )

            send_keys(keys, window_title=game_window, delay=key_press_delay)

            # Step 6: Compute reward and log experience
            if training_enabled:
                reward = compute_reward(prev_state, state, action, config=reward_config)
                episode_reward += reward
                exp = Experience(
                    cycle=cycle,
                    timestamp=time.time(),
                    state=prev_state if prev_state else state,
                    action=action,
                    keys_sent=keys,
                    reward=reward,
                    next_state=state,
                    done=should_stop(state, prev_state),
                )
                replay_buffer.append(exp)
                logger.info(f"Reward: {reward:+.4f} | Episode total: {episode_reward:+.4f}")

            # Step 7: Update previous state for next cycle
            prev_state = state

            # Step 8: Wait
            wait_time = decision_interval
            if should_pause_after(action):
                wait_time = max(wait_time, 1.0)  # extra wait after confirm/cancel

            logger.info(f"Waiting {wait_time}s before next cycle...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        logger.info("Agent interrupted by user. Stopping.")
    except Exception as e:
        logger.error(f"Agent loop error: {e}", exc_info=True)
    finally:
        logger.info("Agent stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Endgame: Singularity AI Agent"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of decision cycles (default: infinite)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single decision cycle and exit",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Export training data and retrain policy from experiences",
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help="Use the learned policy for action selection (if available)",
    )
    parser.add_argument(
        "--reset-policy",
        action="store_true",
        help="Delete the learned policy and start fresh",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Handle policy reset
    if args.reset_policy:
        policy_path = config.get("training", {}).get("policy_path", "screenshots/policy.json")
        p = Path(policy_path)
        if p.exists():
            p.unlink()
            logger.info(f"Deleted policy: {p}")
        return

    # Handle training export
    if args.train:
        from agent.train import export_training_data, train_policy_from_experiences
        replay = ReplayBuffer()
        export_training_data(replay)
        train_policy_from_experiences(replay, output_model_path="screenshots/policy.json")
        print(f"\nPolicy stats: {json.dumps(replay.stats(), indent=2)}")
        return

    # Handle policy-only mode
    if args.policy:
        from agent.train import train_policy_from_experiences
        replay = ReplayBuffer()
        if len(replay.buffer) == 0:
            print("No experiences collected yet. Run the agent first.")
            return
        policy = train_policy_from_experiences(replay, output_model_path="screenshots/policy.json")
        print(f"\nLearned policy ({len(policy)} state patterns):")
        for state_key, entry in policy.items():
            print(f"  {state_key}: {entry['best_action']} (avg_reward={entry['avg_reward']:.2f}, n={entry['count']})")
        return

    if args.once:
        # Single-shot mode for testing
        screenshot_path = get_screenshot_path(0)
        img = capture_full_screen(output_path=screenshot_path)
        if img is None:
            logger.error("No screenshot captured")
            return

        ollama_cfg_data = config.get("ollama", {})
        ollama_config = OllamaConfig(
            host=ollama_cfg_data.get("host", "http://localhost:11434"),
            vision_model=ollama_cfg_data.get("vision_model", "qwen2.5vl"),
        )

        state, state_reasoning = describe_state(screenshot_path, config=ollama_config)
        logger.info(f"State: {json.dumps(state, indent=2)}")
        logger.info(f"Vision reasoning:\n{state_reasoning}")

        action, decision_reasoning = decide_action(state, get_all_actions(), config=ollama_config)
        logger.info(f"Action: {action}")
        logger.info(f"Decision reasoning:\n{decision_reasoning}")
        keys = get_action_keys(action)
        logger.info(f"Keys to press: {keys}")
        send_keys(keys)
    else:
        run_agent_loop(config, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()

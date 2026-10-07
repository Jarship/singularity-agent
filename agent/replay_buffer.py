"""
Experience replay buffer for training the Endgame: Singularity agent.

Logs state-action-reward tuples and provides utilities for:
  - Appending new experiences
  - Computing rewards from state transitions
  - Exporting data for offline fine-tuning
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Experience:
    """A single experience tuple."""

    cycle: int
    timestamp: float
    state: dict  # Game state description from Ollama
    action: str  # Chosen action
    keys_sent: list[str]  # Keys actually pressed
    reward: float  # Computed reward
    next_state: dict | None = None  # State after action
    done: bool = False  # Whether the episode ended (e.g., detected)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "state": self.state,
            "action": self.action,
            "keys_sent": self.keys_sent,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "metadata": self.metadata,
        }


class ReplayBuffer:
    """Stores and manages experience tuples."""

    def __init__(self, max_size: int = 10_000, save_path: str = "screenshots/experiences.jsonl"):
        self.buffer: list[Experience] = []
        self.max_size = max_size
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing experiences if the file exists
        self._load()

    def _load(self) -> None:
        """Load existing experiences from disk."""
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    exp = Experience(
                        cycle=data["cycle"],
                        timestamp=data["timestamp"],
                        state=data["state"],
                        action=data["action"],
                        keys_sent=data["keys_sent"],
                        reward=data["reward"],
                        next_state=data.get("next_state"),
                        done=data.get("done", False),
                        metadata=data.get("metadata", {}),
                    )
                    self.buffer.append(exp)
            print(f"[ReplayBuffer] Loaded {len(self.buffer)} experiences from {self.save_path}")
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"[ReplayBuffer] Failed to load experiences: {e}")

    def append(self, exp: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(exp)
        # Trim to max size (FIFO)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]

        # Append to disk immediately (append-only JSONL)
        with open(self.save_path, "a") as f:
            f.write(json.dumps(exp.to_dict()) + "\n")

    def recent(self, n: int = 50) -> list[Experience]:
        """Get the N most recent experiences."""
        return self.buffer[-n:]

    def filter_action(self, action: str) -> list[Experience]:
        """Get all experiences for a specific action."""
        return [e for e in self.buffer if e.action == action]

    def filter_reward(self, min_reward: float) -> list[Experience]:
        """Get experiences with reward >= min_reward."""
        return [e for e in self.buffer if e.reward >= min_reward]

    def average_reward(self, action: str | None = None) -> float:
        """Compute average reward, optionally filtered by action."""
        experiences = self.buffer if action is None else self.filter_action(action)
        if not experiences:
            return 0.0
        return sum(e.reward for e in experiences) / len(experiences)

    def action_rewards(self) -> dict[str, float]:
        """Average reward per action."""
        rewards: dict[str, float] = {}
        counts: dict[str, int] = {}
        for exp in self.buffer:
            rewards[exp.action] = rewards.get(exp.action, 0.0) + exp.reward
            counts[exp.action] = counts.get(exp.action, 0) + 1
        return {a: rewards[a] / counts[a] for a in rewards}

    def save(self, path: str | None = None) -> None:
        """Save all experiences to a JSONL file."""
        target = Path(path) if path else self.save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            for exp in self.buffer:
                f.write(json.dumps(exp.to_dict()) + "\n")

    def stats(self) -> dict:
        """Return summary statistics."""
        if not self.buffer:
            return {"total": 0}
        rewards = [e.reward for e in self.buffer]
        return {
            "total": len(self.buffer),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "action_counts": {a: len(self.filter_action(a)) for a in set(e.action for e in self.buffer)},
            "action_avg_rewards": self.action_rewards(),
        }

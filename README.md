# Endgame: Singularity — AI Agent

A screenshot-based AI agent that plays **Endgame: Singularity** using a local Ollama vision model and Kev for structured decision-making.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Game Window (Endgame: Singularity, pygame)    │
│  Runs natively on the host                     │
└──────────────────┬────────────────────────────┘
                   │  screenshot (mss) + key injection (xdotool)
                   ▼
┌─────────────────────────────────────────────────┐
│  Docker Container (agent)                      │
│                                                │
│  1. CAPTURE  — mss grabs the game window       │
│  2. DESCRIBE — Ollama vision model (qwen2.5vl) │
│                describes the game state         │
│  3. DECIDE  — Ollama text model or Kev         │
│                chooses the next action          │
│  4. EXECUTE — xdotool sends key presses        │
│  5. WAIT    — pause for natural game break     │
└─────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Ollama Container                              │
│  Runs qwen2.5vl (vision) + text models         │
│  GPU: NVIDIA (NVIDIA Container Toolkit)        │
│  Mac: Metal (automatic with ollama/ollama image)│
└─────────────────────────────────────────────────┘
```

## Prerequisites

### Host Machine

| Requirement | Details |
|---|---|
| **OS** | Linux (X11) or macOS |
| **GPU** | NVIDIA (GTX 1650 Ti or better) for Windows; M2 Mac works for both |
| **Docker** | Docker Engine + Docker Compose |
| **NVIDIA Container Toolkit** | Required for GPU passthrough on Linux |
| **Endgame: Singularity** | Installed on the host (Python 3.9+, pygame, numpy) |

### Install NVIDIA Container Toolkit (Linux)

```bash
# Add the NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Install Endgame: Singularity on the Host

```bash
# Clone the game
git clone https://github.com/singularity/singularity.git
cd singularity

# Install dependencies
pip install pygame numpy polib

# Run the game once to verify it works
python3 -m singularity
```

## Setup

### 1. Clone this repo

```bash
git clone <this-repo>
cd singularity-agent
```

### 2. Pull Ollama models

Start Ollama first, then pull the models:

```bash
docker compose up ollama

# In another terminal, pull the models
docker exec -it singularity-agent-ollama-1 ollama pull qwen2.5vl:2b

# Optional: pull a larger vision model if you have more VRAM
# docker exec -it singularity-agent-ollama-1 ollama pull qwen2.5vl:7b
```

### 3. Configure

Edit `config/config.yaml` to match your setup:

- `display.game_window_title` — the window title of the game (default: `"Endgame: Singularity"`)
- `ollama.host` — Ollama URL (default: `http://ollama:11434` for Docker networking)
- `ollama.vision_model` — model name (default: `qwen2.5vl:2b`)
- `decision.mode` — `"ollama_vision"` (uses Ollama for both vision + decision) or `"kev"` (requires Kev server)

### 4. Run the game on the host

```bash
cd /path/to/singularity
python3 -m singularity
```

**Important:** Set the game to **Borderless Windowed** mode in the in-game options. This ensures the game window is a normal X11 window that the agent can target.

### 5. Run the agent

```bash
# Full agent loop (runs until Ctrl+C)
docker compose up agent

# Single decision cycle (for testing)
docker compose up agent --entrypoint "python -m agent.main --once"

# Limit to N cycles
docker compose up agent --entrypoint "python -m agent.main --max-cycles 50"
```

## File Structure

```
singularity-agent/
├── docker-compose.yml          # Docker Compose orchestration
├── Dockerfile                  # Agent container image
├── config/
│   └── config.yaml             # Agent configuration
├── agent/
│   ├── __init__.py
│   ├── main.py                 # Entry point, main loop
│   ├── capture.py              # Screenshot capture (mss + xdotool)
│   ├── vision.py               # Ollama API integration
│   ├── input.py                # Key injection (xdotool + pyautogui fallback)
│   ├── action_map.py           # High-level action → key sequence mapping
│   └── requirements.txt        # Python dependencies
├── game/                       # Mount point for game data (optional)
├── screenshots/                # Captured screenshots + decision log
└── README.md
```

## How the Agent Decides

1. **Capture** — `mss` grabs the game window region
2. **Describe** — Ollama vision model (`qwen2.5vl:2b`) analyzes the screenshot and returns structured JSON describing the game state (screen type, speed, detection level, resources, available actions)
3. **Decide** — Ollama text model (or Kev) receives the description + available actions and picks the best next action
4. **Execute** — `xdotool` sends the key sequence mapped to that action
5. **Wait** — The agent pauses for a configurable interval before the next cycle

## Training the Agent

The agent includes a built-in training pipeline that improves over time through experience.

### How It Works

1. **Experience Logging** — Every decision cycle logs a `(state, action, reward, next_state)` tuple to `screenshots/experiences.jsonl`
2. **Reward Computation** — A reward function scores each action based on outcomes:
   - Detection increased → negative reward
   - Research completed → positive reward
   - Base built → positive reward
   - Money earned → small positive reward
   - Survived a cycle → small positive reward
3. **Policy Learning** — A simple frequency-based policy maps state patterns to the best-performing actions
4. **Prompt Improvement** — After 10+ experiences, the decision prompt is enhanced with historical success rates

### Training Commands

```bash
# Run the agent normally (training is on by default)
docker compose up agent

# After some gameplay, export training data and build a policy
docker compose up agent --entrypoint "python -m agent.main --train"

# Inspect the learned policy
docker compose up agent --entrypoint "python -m agent.main --policy"

# Reset the learned policy and start fresh
docker compose up agent --entrypoint "python -m agent.main --reset-policy"

# Run a single cycle (for testing)
docker compose up agent --entrypoint "python -m agent.main --once"
```

### Tuning Rewards

Edit `config/config.yaml` → `reward:` section to adjust behavior:

| Parameter | Effect |
|---|---|
| `detection_critical_penalty` | How hard the agent avoids being detected |
| `research_completed_reward` | How much the agent values finishing research |
| `base_built_reward` | How much the agent values building new bases |
| `speed_max_reward` | How much the agent values playing fast |

### From Simple to Advanced Training

| Level | Approach | What it does |
|---|---|---|
| **1. Reward shaping** (implemented) | Reward function + policy cache | Agent learns which actions work in which states |
| **2. Prompt improvement** (implemented) | Historical success rates in decision prompt | LLM makes better decisions using past experience |
| **3. Offline fine-tuning** (future) | Export experiences → fine-tune Kev/VEV with LoRA | Model learns from its own gameplay data |
| **4. Online RL** (future) | PPO/DQN on top of the LLM's decisions | Agent learns a policy that overrides the LLM when it knows better |

### Level 3: Offline Fine-tuning (when you have enough data)

Once you've collected 500+ experiences, you can fine-tune a small model:

```bash
# Export training data
docker compose up agent --entrypoint "python -m agent.main --train"

# The training data is in screenshots/training_data.json
# You can then fine-tune Kev or VEV using the exported data:
# - Use QLoRA on Kev-4B with the experience data
# - Or use VEV's training pipeline with screenshot+question pairs
```

### Level 4: Online RL (advanced)

For real-time learning during gameplay, you'd add a small policy network
that learns to override the LLM's decisions based on reward signals.
This requires a framework like `stable-baselines3` or `rlax` and is
not yet implemented.

| Action | Keys | Description |
|---|---|---|
| `speed_max` | `4` | Set game speed to max |
| `speed_pause` | `0` | Pause the game |
| `speed_slow` | `1` | Set game speed to slow |
| `speed_toggle` | `Space` | Toggle pause/unpause |
| `confirm` | `Return` | Confirm choice |
| `cancel` | `Escape` | Cancel dialog |
| `cancel_twice` | `Escape, Escape` | Back to map screen |
| `research_next` | `Tab` | Next research option |
| `research_confirm` | `Return` | Confirm research |
| `build_base` | `B` | Open build menu |
| `abort_mission` | `Escape, Escape` | Return to map |
| `emergency_pause` | `0` | Pause when detection is critical |

## Hardware Notes

### M2 MacBook Pro (16GB unified memory)
- Can run `qwen2.5vl:7b` + `kev-4b` comfortably
- Ollama uses Metal acceleration automatically
- Game runs natively on macOS

### Windows Laptop (GTX 1650 Ti Max-Q, ~4GB VRAM)
- Use `qwen2.5vl:2b` for vision (fits in 4GB)
- Use `kev-0.8b` for decisions (fits in 4GB)
- Run game in borderless windowed mode
- Requires NVIDIA Container Toolkit in WSL2

## Troubleshooting

### Game window not found
- Verify the game is running in **borderless windowed** mode
- Check the window title with `xdotool search --name "Endgame"`
- Update `display.game_window_title` in `config/config.yaml`

### Ollama connection refused
- Ensure the Ollama container is running: `docker compose ps`
- Check Ollama logs: `docker compose logs ollama`
- Pull the model first: `docker exec -it <ollama-container> ollama pull qwen2.5vl:2b`

### Keys not reaching the game
- The game must be in **borderless windowed** mode (not fullscreen exclusive)
- Try increasing `timing.key_press_delay` in config
- On macOS, `pyautogui` may need Accessibility permissions

### Screenshots are black
- X11 display not shared — ensure `- /tmp/.X11-unix:/tmp/.X11-unix` is in volumes
- On Wayland, you may need `xwayland` or a different capture approach

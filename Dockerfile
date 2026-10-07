FROM python:3.12-slim

# System deps for Linux/X11 environments only.
# On macOS, the agent runs natively (no Docker needed for the agent).
# xdotool and x11-utils require X11 and are only needed on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    xdotool \
    x11-utils \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir mss

# Copy agent code
COPY agent/ /app/agent/
COPY config/ /app/config/

WORKDIR /app

# Default: run the agent loop
CMD ["python", "-m", "agent.main"]

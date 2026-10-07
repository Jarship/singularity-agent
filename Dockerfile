FROM python:3.12-bookworm-slim

# System deps: X11 display, xdotool for key injection, mss for screenshots
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

# Copy agent code
COPY agent/ /app/agent/
COPY config/ /app/config/

WORKDIR /app

# Default: run the agent loop
CMD ["python", "-m", "agent.main"]

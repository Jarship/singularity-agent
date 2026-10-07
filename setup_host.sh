#!/usr/bin/env bash
# setup_host.sh — Install Endgame: Singularity on the host machine
# Run this on the machine where the game will run (not in Docker).

set -euo pipefail

echo "=== Endgame: Singularity Host Setup ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Installing Python 3..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip
    elif command -v brew &>/dev/null; then
        brew install python
    else
        echo "ERROR: Cannot install Python automatically on this OS."
        echo "Install Python 3.9+ manually and retry."
        exit 1
    fi
fi

python3 --version

# Clone the game
if [ ! -d "singularity" ]; then
    echo "Cloning Endgame: Singularity..."
    git clone https://github.com/singularity/singularity.git
fi

cd singularity

# Install Python dependencies
echo "Installing Python dependencies..."
pip install pygame numpy polib

echo ""
echo "=== Setup Complete ==="
echo "Run the game with: python3 -m singularity"
echo ""
echo "IMPORTANT: Set the game to Borderless Windowed mode in the options."
echo "The AI agent needs the game window to be a normal X11 window."

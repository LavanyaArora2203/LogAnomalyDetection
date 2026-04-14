#!/bin/bash
#
# Copyright (c) 2025 Omer Zak
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Docker entrypoint for Google Antigravity
# Runs as root to fix volume permissions, then drops to ubuntu user (1000:1000)

# Ensure directories exist and fix ownership on mounted volumes
mkdir -p /home/ubuntu/.config/Antigravity \
         /home/ubuntu/.antigravity/extensions \
         /home/ubuntu/.config/google-chrome \
         /home/ubuntu/.local/share/keyrings \
         /home/ubuntu/.cache \
         /tmp/runtime

chown -R 1000:1000 /home/ubuntu /tmp/runtime
chmod 700 /tmp/runtime

# Start openbox window manager in background for window decorations
if [ -n "$DISPLAY" ] && command -v openbox >/dev/null 2>&1; then
    gosu 1000:1000 openbox &
fi

# Drop to ubuntu user and execute command
exec gosu 1000:1000 "$@"

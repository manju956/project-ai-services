#!/bin/sh
# entrypoint.sh — run as root, fix SSH key permissions, then drop to UID 1001.
#
# Podman/Docker mounts files owned by the host user (often root:root, mode 600).
# The container process runs as UID 1001 and cannot read a root-owned 600 file.
# This script copies the key to a temp path owned by UID 1001 before exec-ing.
#
# Uses 'runuser' from shadow-utils (already installed in the image) — no extra
# packages required.

set -e

if [ -n "$IBMI_SSH_KEY_PATH" ] && [ -f "$IBMI_SSH_KEY_PATH" ]; then
    _SAFE_KEY="/tmp/ibmi_key_$$"
    cp "$IBMI_SSH_KEY_PATH" "$_SAFE_KEY"
    chown 1001:0 "$_SAFE_KEY"
    chmod 600 "$_SAFE_KEY"
    export IBMI_SSH_KEY_PATH="$_SAFE_KEY"
fi

exec runuser -u appuser -- "$@"

#!/bin/sh
# entrypoint.sh — fix permissions on mounted volumes, then drop to UID 1001.
#
# Mounted directories (e.g. /models for the Triton repo) are created by the
# host as root:root.  This script chowns them to UID 1001 before exec-ing
# the requested command as appuser.

set -e

# Fix write permission on the Triton model repository output directory
if [ -n "$TRITON_REPO" ] && [ -d "$TRITON_REPO" ]; then
    chown -R 1001:0 "$TRITON_REPO"
fi

exec runuser -u appuser -- "$@"

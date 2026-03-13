#!/bin/bash
set -e

# Create workspace dirs if they don't exist (runs as root here)
mkdir -p /workspace/data /workspace/runs /workspace/mlruns /workspace/models /workspace/.yolo /workspace/.matplotlib

# Drop to host user if HOST_UID is provided (and it is not root)
if [ -n "${HOST_UID}" ] && [ "${HOST_UID}" != "0" ]; then
    HOST_GID="${HOST_GID:-${HOST_UID}}"

    # Create group/user if they don't exist in the container
    # --home /workspace ensures gosu sets HOME=/workspace (not /home/hostuser)
    getent group "${HOST_GID}" >/dev/null 2>&1 || groupadd --gid "${HOST_GID}" hostgroup
    getent passwd "${HOST_UID}" >/dev/null 2>&1 || useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --no-create-home --home /workspace --shell /bin/bash hostuser

    # Only chown the writable dirs — NOT /workspace itself (may contain :ro mounts)
    chown -R "${HOST_UID}:${HOST_GID}" /workspace/data /workspace/runs /workspace/mlruns /workspace/models /workspace/.yolo /workspace/.matplotlib

    export YOLO_CONFIG_DIR=/workspace/.yolo
    export MPLCONFIGDIR=/workspace/.matplotlib

    exec gosu "${HOST_UID}:${HOST_GID}" yolo-boost "$@"
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/workspace/.yolo}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.matplotlib}"
exec yolo-boost "$@"

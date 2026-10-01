#!/bin/sh
# Prepare the data volume, then exec the app (exec keeps signals reaching uvicorn).
set -eu

APP_UID=10001
APP_GID=10001
DATA_DIR="${DATA_DIR:-/data}"
export UVICORN_PORT="${PORT:-8000}"

if [ "$(id -u)" = "0" ]; then
    # Volumes on Fly.io (and fresh bind mounts) are root-owned: hand them to the app user.
    mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/rag_store"
    if [ "$(stat -c %u "$DATA_DIR")" != "$APP_UID" ]; then
        echo "entrypoint: chowning $DATA_DIR to $APP_UID:$APP_GID" >&2
        chown -R "$APP_UID:$APP_GID" "$DATA_DIR"
    fi
    exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --clear-groups --no-new-privs -- "$@"
fi

if [ ! -w "$DATA_DIR" ]; then
    echo "entrypoint: $DATA_DIR is not writable by uid $(id -u). Start the container as root once" \
         "(it will chown the volume) or run: chown -R $APP_UID:$APP_GID <volume path>" >&2
    exit 1
fi
mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/rag_store"
exec "$@"

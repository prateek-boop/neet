#!/bin/sh
# Runs as root: bind mounts and named volumes (./index_store, ./uploads, the
# hf-cache volume) frequently come up root-owned on first boot, so the app's
# unprivileged user couldn't write to them. Fix ownership on the writable paths,
# then drop to that user for the actual process. /app/data is mounted read-only
# and is deliberately NOT chowned (the app only reads from it).
set -e

APP_USER="${APP_USER:-app}"

for d in /app/index_store /app/uploads /app/.cache/huggingface; do
    if [ -d "$d" ]; then
        chown -R "$APP_USER:$APP_USER" "$d" 2>/dev/null \
            || echo "entrypoint: warning — could not chown $d (continuing; check host perms)"
    fi
done

# Drop root and exec the CMD (uvicorn) as the unprivileged user.
exec gosu "$APP_USER" "$@"

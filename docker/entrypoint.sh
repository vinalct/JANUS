#!/bin/sh

set -e

# Give an arbitrary Docker uid a per-process account view without changing system files.
# Podman keep-id and the image's uid 1000 already resolve, so they skip this branch.
if ! whoami >/dev/null 2>&1; then
    export NSS_WRAPPER_PASSWD=/tmp/janus-passwd NSS_WRAPPER_GROUP=/tmp/janus-group
    printf 'janus:x:%s:%s:janus:/tmp:/usr/sbin/nologin\n' "$(id -u)" "$(id -g)" > "$NSS_WRAPPER_PASSWD"
    printf 'janus:x:%s:\n' "$(id -g)" > "$NSS_WRAPPER_GROUP"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnss_wrapper.so
fi

exec "$@"

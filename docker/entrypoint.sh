#!/bin/sh

set -e

# Give an arbitrary Docker uid a per-process account view without changing system files.
# Podman keep-id and the image's uid 1000 already resolve, so they skip this branch.
if ! whoami >/dev/null 2>&1; then
    NSS_WRAPPER_PASSWD="$(mktemp /tmp/janus-passwd.XXXXXX)"
    NSS_WRAPPER_GROUP="$(mktemp /tmp/janus-group.XXXXXX)"
    export NSS_WRAPPER_PASSWD NSS_WRAPPER_GROUP
    printf 'janus:x:%s:%s:janus:/tmp:/usr/sbin/nologin\n' "$(id -u)" "$(id -g)" > "$NSS_WRAPPER_PASSWD"
    printf 'janus:x:%s:\n' "$(id -g)" > "$NSS_WRAPPER_GROUP"
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnss_wrapper.so
fi

exec "$@"

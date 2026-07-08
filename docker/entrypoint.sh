#!/bin/sh

set -e

if ! whoami >/dev/null 2>&1; then
    echo "janus:x:$(id -u):$(id -g):janus:/tmp:/usr/sbin/nologin" >> /etc/passwd
fi

exec "$@"

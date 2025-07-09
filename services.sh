#!/usr/bin/env bash

if [ "$1" != "start" ] && [ "$1" != "stop" ] && [ "$1" != "restart" ]; then
    echo "Usage: $0 {start|stop|restart}"
    exit 1
fi

systemctl --user $1 tools-api.service
systemctl --user $1 supervisor.service
systemctl --user $1 proxy-router.service
systemctl --user $1 agent.service

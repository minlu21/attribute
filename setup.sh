#!/bin/bash

workdir=$(pwd)
# image=...

if ! id -nzG | grep -qzxF docker; then
    exec sudo -g docker "$0" "$@"
fi

mkdir -p "$workdir"

if [[ $1 = --pull ]] || ! docker image inspect "$image" &>/dev/null; then
    docker pull "$image"
fi



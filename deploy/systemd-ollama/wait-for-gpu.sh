#!/bin/sh
# Gate for ollama.service: wait until the GPU is usable before Ollama
# starts its llama-server GPU discovery.
#
# Why: on WSL2 Ollama can start before the GPU stack is ready. Its GPU
# probe has a 30 s watchdog per backend; when the probes time out, Ollama
# registers a CPU-only "inference compute" for the whole process lifetime
# and every embedding runs on CPU (minutes per batch instead of
# milliseconds).
#
# Bounded wait: at most 60 s, then start anyway (never block boot forever).
# Exit code is always 0, so a missing or broken nvidia-smi cannot prevent
# the service from starting; a failed discovery is repaired later by
# ollama-gpu-guard.timer.

set -u

is_wsl=no
if grep -qi microsoft /proc/version 2>/dev/null; then
    is_wsl=yes
fi

i=0
while [ "$i" -lt 60 ]; do
    smi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        smi=$(command -v nvidia-smi)
    elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
        smi=/usr/lib/wsl/lib/nvidia-smi
    fi
    if [ "$is_wsl" = "yes" ] && [ ! -e /dev/dxg ]; then
        # WSL GPU device not exported yet: keep waiting.
        i=$((i + 1))
        sleep 1
        continue
    fi
    if [ -z "$smi" ]; then
        # /dev/dxg is present but the CUDA userspace (and nvidia-smi with
        # it) is still being mounted: keep waiting on WSL. On a host with
        # no NVIDIA tooling at all there is nothing to wait for.
        [ "$is_wsl" = "yes" ] || exit 0
        i=$((i + 1))
        sleep 1
        continue
    fi
    if "$smi" -L >/dev/null 2>&1; then
        exit 0
    fi
    i=$((i + 1))
    sleep 1
done
exit 0

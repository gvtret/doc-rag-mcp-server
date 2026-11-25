#!/bin/sh
# One-shot post-boot self-heal for Ollama.
#
# If this boot's journal shows that the running Ollama registered CPU-only
# inference, restart it once. The GPU must be verified usable first; a
# restart on a host where the GPU is still missing would not help and
# would only add churn.
#
# Idempotent: /run/ollama-gpu-guard.done lives on tmpfs and disappears on
# reboot, so at most one repair attempt happens per boot. Exit code is
# always 0: the guard must never fail the timer nor restart in a loop.
#
# The journal is filtered through grep and only the last matching line is
# kept, so a verbose (verbosity 4) Ollama log cannot grow shell memory.

set -u

marker=/run/ollama-gpu-guard.done
[ -e "$marker" ] && exit 0

last_compute=$(journalctl -u ollama -b --no-pager 2>/dev/null | grep 'inference compute' | tail -n 1)
if [ -n "$last_compute" ]; then
    if ! printf '%s\n' "$last_compute" | grep -qi 'library=cpu'; then
        exit 0 # a GPU backend is in use -> nothing to repair
    fi
else
    # No model has been loaded this boot, so there is no "inference
    # compute" line yet. Fall back to discovery failures as the signal.
    fails=$(journalctl -u ollama -b --no-pager 2>/dev/null \
        | grep -Ec 'failure during llama-server GPU discovery|GPU discovery watchdog timed out')
    [ "$fails" -gt 0 ] || exit 0
fi

# Verify the GPU is actually usable now; if not, do not restart (it would
# come up CPU-only again and mask the real problem).
if [ -x /usr/local/sbin/wait-for-gpu.sh ]; then
    /usr/local/sbin/wait-for-gpu.sh || exit 0
fi
smi=""
if command -v nvidia-smi >/dev/null 2>&1; then
    smi=$(command -v nvidia-smi)
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    smi=/usr/lib/wsl/lib/nvidia-smi
fi
if [ -z "$smi" ] || ! "$smi" -L >/dev/null 2>&1; then
    logger -t ollama-gpu-guard "GPU is not usable; skipping the restart"
    exit 0
fi

if systemctl restart ollama; then
    : > "$marker"
    logger -t ollama-gpu-guard "Ollama was CPU-only; restarted it once to restore GPU inference"
else
    logger -t ollama-gpu-guard -p user.err "systemctl restart ollama failed; manual intervention needed"
    exit 0
fi
exit 0

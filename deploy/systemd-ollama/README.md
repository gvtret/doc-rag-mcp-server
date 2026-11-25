# Защита Ollama от CPU-фолбэка (WSL2)

Комплект системных юнитов и скриптов, которые не дают Ollama остаться
без GPU после загрузки WSL2. Без них `ollama.service` может стартовать
раньше готовности GPU: оба пробы CUDA-дискавери упираются в 30-секундный
watchdog, и процесс на всё время жизни регистрирует CPU-only backend
(`inference compute id=cpu library=cpu`). Эмбеддинги bge-m3 в этом
режиме идут минутами, и MCP-вызовы doc-rag падают по таймауту (504).

## Состав

| Файл | Куда ставится | Назначение |
|---|---|---|
| `wait-for-gpu.sh` | `/usr/local/sbin/wait-for-gpu.sh` | Ждёт до 60 с готовности `/dev/dxg` и NVML (`nvidia-smi -L`); всегда завершается с кодом 0 |
| `ollama-gpu-ready.conf` | `/etc/systemd/system/ollama.service.d/gpu-ready.conf` | `ExecStartPre` для `ollama.service`: запускает гейт |
| `keepalive.conf` | `/etc/systemd/system/ollama.service.d/keepalive.conf` | `OLLAMA_KEEP_ALIVE=30m`, чтобы модель не выгружалась между запросами |
| `ollama-gpu-guard.sh` | `/usr/local/sbin/ollama-gpu-guard.sh` | Одноразовая самопроверка: если текущий boot всё же CPU-only, перезапускает Ollama |
| `ollama-gpu-guard.service` | `/etc/systemd/system/ollama-gpu-guard.service` | oneshot-юнит для скрипта |
| `ollama-gpu-guard.timer` | `/etc/systemd/system/ollama-gpu-guard.timer` | Запуск через 90 с после boot |

## Установка

```bash
sudo install -D -m 0755 -o root -g root deploy/systemd-ollama/wait-for-gpu.sh /usr/local/sbin/wait-for-gpu.sh
sudo install -D -m 0755 -o root -g root deploy/systemd-ollama/ollama-gpu-guard.sh /usr/local/sbin/ollama-gpu-guard.sh
sudo install -D -m 0644 -o root -g root deploy/systemd-ollama/ollama-gpu-ready.conf /etc/systemd/system/ollama.service.d/gpu-ready.conf
sudo install -D -m 0644 -o root -g root deploy/systemd-ollama/ollama-gpu-guard.service /etc/systemd/system/ollama-gpu-guard.service
sudo install -D -m 0644 -o root -g root deploy/systemd-ollama/ollama-gpu-guard.timer /etc/systemd/system/ollama-gpu-guard.timer
sudo install -D -m 0644 -o root -g root deploy/systemd-ollama/keepalive.conf /etc/systemd/system/ollama.service.d/keepalive.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable --now ollama-gpu-guard.timer
```

## Проверка

```bash
systemctl show ollama -p ExecStartPre     # должен показать /usr/local/sbin/wait-for-gpu.sh
curl -s http://127.0.0.1:11434/api/embed -d '{"model":"bge-m3","input":"ok"}' >/dev/null
ollama ps                                  # PROCESSOR: 100% GPU
curl -s http://127.0.0.1:3333/health       # "ollama_backend":"gpu"
systemctl list-timers ollama-gpu-guard.timer
```

Поведение при проблемах: `wait-for-gpu.sh` ждёт готовности GPU до 60 с и
завершается успешно в любом случае; на хосте без NVIDIA-тулинга (и не в
WSL) он выходит сразу. `ollama-gpu-guard` перезапускает Ollama только если
GPU уже доступна; если нет - пропускает рестарт и пишет в журнал, чтобы не
маскировать проблему.

## Откат

```bash
sudo systemctl disable --now ollama-gpu-guard.timer 2>/dev/null || true
sudo rm -f /etc/systemd/system/ollama.service.d/gpu-ready.conf \
          /etc/systemd/system/ollama-gpu-guard.service \
          /etc/systemd/system/ollama-gpu-guard.timer \
          /usr/local/sbin/wait-for-gpu.sh /usr/local/sbin/ollama-gpu-guard.sh
sudo systemctl daemon-reload
```

`keepalive.conf` можно оставить или вернуть `OLLAMA_KEEP_ALIVE=1m`.

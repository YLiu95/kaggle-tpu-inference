#!/usr/bin/env python3
"""Persistent inference daemon.

Loads and shards the weights once, compiles the prefill/decode programs once, and then
serves streaming generations over a Unix socket so repeat runs start in milliseconds
instead of ~3.5 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback

os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("HF_HOME", "/root/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax

CACHE_DIR = os.environ.get("GEMMA4_XLA_CACHE", "/root/.cache/gemma4_jax")
os.makedirs(CACHE_DIR, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_tpu.engine import Engine  # noqa: E402
from gemma4_tpu.session import (  # noqa: E402
    SOCKET_PATH,
    generate_events,
    model_info,
    resolve_model_dir,
)
from gemma4_tpu.tpu_monitor import TpuMonitor  # noqa: E402

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"


class Daemon:
    def __init__(self, model_id: str, model_dir: str, max_len: int, top_k: int, path: str):
        self.model_id = model_id
        self.path = path
        self.lock = threading.Lock()
        self.requests = 0
        self.started = time.time()
        self._stop = threading.Event()

        log(f"loading {model_id}")
        t0 = time.time()
        self.engine = Engine(model_dir, max_len=max_len, top_k=top_k)
        log(f"weights sharded across {self.engine.n_devices} chips in {time.time() - t0:.1f}s")

        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)

        log("compiling prefill + decode")
        ct = self.engine.compile_all()
        log(
            f"compiled prefill {ct['prefill_compile_s']:.1f}s decode "
            f"{ct['decode_compile_s']:.1f}s, decode step {1000 * ct['decode_step_s']:.2f} ms"
        )
        self.monitor = TpuMonitor()
        self.monitor.start()

    def info(self) -> dict:
        return model_info(
            self.engine, self.model_id,
            served_by=f"persistent daemon pid {os.getpid()} ({self.path})",
        )

    def status(self) -> dict:
        return {
            "kind": "status",
            "pid": os.getpid(),
            "model": self.model_id,
            "uptime_s": round(time.time() - self.started, 1),
            "requests": self.requests,
            "busy": self.lock.locked(),
            "max_len": self.engine.max_len,
            "decode_step_ms": round(1000 * self.engine.decode_step_seconds, 2),
        }

    # ------------------------------------------------------------------ serving
    def handle(self, conn: socket.socket):
        conn.settimeout(600.0)
        with conn, conn.makefile("rw") as f:
            line = f.readline()
            if not line.strip():
                return
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                f.write(json.dumps({"kind": "pong"}) + "\n")
                f.flush()
                return
            if cmd == "status":
                f.write(json.dumps(self.status()) + "\n")
                f.flush()
                return
            if cmd == "shutdown":
                f.write(json.dumps({"kind": "bye"}) + "\n")
                f.flush()
                self._stop.set()
                return

            if not self.lock.acquire(blocking=False):
                f.write(json.dumps({"kind": "error", "message": "daemon busy"}) + "\n")
                f.write(json.dumps({"kind": "done"}) + "\n")
                f.flush()
                return
            try:
                self.requests += 1
                for ev in generate_events(self.engine, self.tok, self.monitor, req, self.info()):
                    f.write(json.dumps(ev) + "\n")
                    f.flush()
            except (BrokenPipeError, ConnectionResetError):
                log("client disconnected mid-generation")
            except Exception:
                log(traceback.format_exc())
                try:
                    f.write(json.dumps({"kind": "error", "message": "generation failed"}) + "\n")
                    f.write(json.dumps({"kind": "done"}) + "\n")
                    f.flush()
                except OSError:
                    pass
            finally:
                self.lock.release()

    def serve_forever(self):
        if os.path.exists(self.path):
            os.unlink(self.path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        os.chmod(self.path, 0o600)
        srv.listen(8)
        srv.settimeout(1.0)
        log(f"READY listening on {self.path}")
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        finally:
            srv.close()
            if os.path.exists(self.path):
                os.unlink(self.path)
            self.monitor.stop()
            log("stopped")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--socket", default=SOCKET_PATH)
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model, args.model_dir)
    daemon = Daemon(args.model, model_dir, args.max_len, args.top_k, args.socket)
    daemon.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

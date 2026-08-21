"""Unified event stream: either produced locally or read from the persistent daemon."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Iterator

from .limits import MAX_OUTPUT_TOKENS, check_completion, check_request
from .models import try_resolve

SOCKET_PATH = os.environ.get("GEMMA4_SOCKET", "/tmp/gemma4-tpu.sock")
TPU_SNAPSHOT_INTERVAL = 0.25


def resolve_model_dir(model_id: str, local_dir: str | None = None) -> str:
    if local_dir:
        return local_dir
    env = os.environ.get("GEMMA4_MODEL_DIR")
    if env and os.path.isdir(env):
        return env
    spec = try_resolve(model_id)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        spec.repo_id if spec else model_id,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        max_workers=8,
    )


def build_prompt_ids(tok, prompt: str, system: str | None, think: bool) -> list[int]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    encoded = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=think,
        tokenize=True, return_dict=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def kv_bytes_per_token(cfg) -> int:
    return sum(2 * 2 * cfg.kv_heads_for(lt) * cfg.head_dim_for(lt) for lt in cfg.layer_types)


def model_info(engine, model_id: str, served_by: str | None = None) -> dict:
    cfg = engine.cfg
    spec = try_resolve(model_id)
    return {
        "kind": "info",
        "model": spec.repo_id if spec else model_id,
        "model_key": spec.key if spec else None,
        "model_label": spec.label if spec else model_id,
        "model_kind": "moe" if cfg.enable_moe_block else "dense",
        "model_max_context": spec.max_context if spec else engine.max_len,
        "param_bytes": engine.param_bytes,
        "active_params": engine.active_params,
        "cache_bytes": engine.cache_bytes,
        "max_len": engine.max_len,
        "n_devices": engine.n_devices,
        "num_layers": cfg.num_hidden_layers,
        "sliding_layers": sum(1 for x in cfg.layer_types if x == "sliding_attention"),
        "full_layers": sum(1 for x in cfg.layer_types if x == "full_attention"),
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "num_experts": cfg.num_experts,
        "top_k_experts": cfg.top_k_experts,
        "kv_bytes_per_token": kv_bytes_per_token(cfg),
        "hbm_per_chip_bytes": engine.hbm["total_bytes_per_chip"],
        "decode_step_s": engine.decode_step_seconds,
        "max_output_tokens": engine.max_output_tokens,
        "served_by": served_by,
    }


def marker_ids(tok) -> tuple[int | None, int | None]:
    def one(text: str) -> int | None:
        ids = tok.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    return one("<|channel>"), one("<channel|>")


def generate_events(engine, tok, monitor, request: dict, info: dict) -> Iterator[dict]:
    """Core generator shared by the local path and the daemon."""
    from .stream import ChannelSplitter, IncrementalDetokenizer

    prompt_ids = build_prompt_ids(
        tok, request["prompt"], request.get("system"), request.get("think", True)
    )
    requested = max(1, int(request.get("max_new_tokens", 768)))
    max_output = getattr(engine, "max_output_tokens", MAX_OUTPUT_TOKENS)
    allowed, warnings = check_request(
        requested,
        len(prompt_ids),
        engine.max_len,
        max_output,
        model_max_context=info.get("model_max_context"),
    )
    info = {
        **info,
        "requested_max_new_tokens": requested,
        "allowed_max_new_tokens": allowed,
        "max_output_tokens": max_output,
    }
    yield info
    yield {"kind": "prompt", "text": request["prompt"], "tokens": len(prompt_ids)}
    for w in warnings:
        yield w.to_event()
    if allowed < 1:
        yield {"kind": "done"}
        return

    detok = IncrementalDetokenizer(tok)
    splitter = ChannelSplitter(*marker_ids(tok))
    stop_ids = set(engine.cfg.eos_token_ids)
    last_snap = 0.0
    finish = {"generated": 0, "stopped_naturally": True, "allowed": allowed}

    for ev in engine.generate(
        prompt_ids,
        max_new_tokens=allowed,
        temperature=float(request.get("temperature", 1.0)),
        top_p=float(request.get("top_p", 0.95)),
        seed=int(request.get("seed", 0)),
    ):
        if ev["kind"] == "finish":
            finish = ev
            continue
        now = time.perf_counter()
        if monitor is not None and now - last_snap > TPU_SNAPSHOT_INTERVAL:
            yield {"kind": "tpu", **monitor.latest.to_dict()}
            last_snap = now

        text = ""
        if ev["token"] not in stop_ids:
            parts = splitter.feed_token(ev["token"], detok.add(ev["token"]))
            text = "".join(chunk for _, chunk in parts)
        out = {
            "kind": "prefill" if ev["kind"] == "prefill" else "token",
            "t": ev["t"],
            "text": text,
            "mode": splitter.mode,
        }
        if ev["kind"] == "prefill":
            out["ttft"] = ev["ttft"]
            out["prompt_tokens"] = ev["prompt_tokens"]
        yield out

    for mode, chunk in splitter.flush():
        yield {"kind": "token", "t": time.perf_counter(), "text": chunk, "mode": mode}
    for w in check_completion(
        finish["generated"], finish["allowed"], finish["stopped_naturally"]
    ):
        yield w.to_event()
    if monitor is not None:
        yield {"kind": "tpu", **monitor.latest.to_dict()}
    yield {"kind": "done"}


# ------------------------------------------------------------------------ remote client
def daemon_alive(path: str = SOCKET_PATH, timeout: float = 2.0) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            s.sendall(json.dumps({"cmd": "ping"}).encode() + b"\n")
            return bool(s.makefile("r").readline())
    except OSError:
        return False


def remote_events(request: dict, path: str = SOCKET_PATH) -> Iterator[dict]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(600.0)
        s.connect(path)
        s.sendall(json.dumps(request).encode() + b"\n")
        with s.makefile("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                yield ev
                if ev["kind"] == "done":
                    return


def send_command(cmd: str, path: str = SOCKET_PATH, timeout: float = 5.0) -> dict | None:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            s.sendall(json.dumps({"cmd": cmd}).encode() + b"\n")
            line = s.makefile("r").readline()
            return json.loads(line) if line.strip() else None
    except OSError:
        return None

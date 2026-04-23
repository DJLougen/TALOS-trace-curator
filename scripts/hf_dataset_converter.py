#!/usr/bin/env python3
"""
hf_dataset_converter.py
Converts external HF datasets (clem/hf-coding-tools-traces, badlogicgames/pi-mono)
into Hermes-compatible session traces for trace-curator processing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def convert_hf_coding_tools(input_dir: str, output_file: str) -> int:
    """
    Convert clem/hf-coding-tools-traces event-level JSONL into session-level traces.
    Input: each line is {sessionId, type, message: {role, content}, benchmark_metadata: {tool, effort, thinking, cost, latency}}
    Output: each line is a trace {session_id, messages: [{role, content}], harness, effort, thinking_enabled}
    """
    in_path = Path(input_dir)
    events: Dict[str, List[Dict[str, Any]]] = {}
    session_meta: Dict[str, Dict[str, Any]] = {}

    jsonl_files = list(in_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = obj.get("sessionId")
                if not sid:
                    continue
                if sid not in events:
                    events[sid] = []
                    bm = obj.get("benchmark_metadata", {})
                    session_meta[sid] = {
                        "harness": bm.get("tool", "unknown"),
                        "effort": bm.get("effort", "unknown"),
                        "thinking_enabled": bm.get("thinking", False),
                    }
                events[sid].append(obj)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, evs in events.items():
            evs.sort(key=lambda x: x.get("timestamp", x.get("id", "")))
            messages = []
            for ev in evs:
                msg = ev.get("message", {})
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    messages.append({"role": role, "content": content})
            if len(messages) < 2:
                continue
            trace = {
                "session_id": sid,
                "messages": messages,
                "harness": session_meta[sid]["harness"],
                "effort": session_meta[sid]["effort"],
                "thinking_enabled": session_meta[sid]["thinking_enabled"],
                "source": "clem/hf-coding-tools-traces",
            }
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} session traces to {output_file}")
    return written


def convert_pi_mono(input_dir: str, output_file: str) -> int:
    """
    Convert badlogicgames/pi-mono file-per-session JSONL into session-level traces.
    Input: each file is a session with lines like {type: "message", message: {role, content: [{type, text|thinking|toolCall}]}}
    Output: each line is a trace {session_id, messages: [{role, content}], thinking: str, harness}
    """
    in_path = Path(input_dir)
    jsonl_files = list(in_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {input_dir}")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for fpath in jsonl_files:
            with open(fpath, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            session_id = None
            messages: List[Dict[str, str]] = []
            thinking_parts: List[str] = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("type") == "session":
                    session_id = obj.get("id")
                    continue

                if obj.get("type") != "message":
                    continue

                msg = obj.get("message", {})
                role = msg.get("role", "")
                content_list = msg.get("content", [])

                # Flatten content array
                texts = []
                for c in content_list:
                    ctype = c.get("type")
                    if ctype == "text":
                        texts.append(c.get("text", ""))
                    elif ctype == "thinking":
                        thinking_parts.append(c.get("thinking", ""))
                    elif ctype == "toolCall":
                        # Represent tool calls as markdown-like blocks
                        name = c.get("name", "")
                        args = json.dumps(c.get("arguments", {}))
                        texts.append(f"<tool_call>\n{name}: {args}\n</tool_call>")
                    elif ctype == "text" and role == "toolResult":
                        texts.append(c.get("text", ""))

                full_text = "\n".join(texts).strip()
                if role and full_text:
                    # Map pi-mono roles to standard roles
                    if role == "toolResult":
                        # toolResult is the result of a tool call, map to a generic assistant/tool result
                        messages.append({"role": "tool", "content": full_text})
                    else:
                        messages.append({"role": role, "content": full_text})

            if len(messages) < 2:
                continue

            if not session_id:
                session_id = fpath.stem

            trace = {
                "session_id": session_id,
                "messages": messages,
                "thinking": "\n\n".join(thinking_parts).strip(),
                "harness": "pi-mono",
                "source": "badlogicgames/pi-mono",
            }
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} session traces to {output_file}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert external HF datasets to Hermes trace format")
    parser.add_argument("--dataset", choices=["hf-coding-tools", "pi-mono"], required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "hf-coding-tools":
        convert_hf_coding_tools(args.input_dir, args.output_file)
    elif args.dataset == "pi-mono":
        convert_pi_mono(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()

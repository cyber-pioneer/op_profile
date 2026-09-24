#!/usr/bin/env python3
"""Reproduce missing API attribution for an Inductor kernel on CUDA Graph replay.

This file only requires a CUDA-enabled PyTorch installation with torch.compile.
It deliberately profiles one direct compiled call and several graph replays of
the same kernel, so both attribution paths can be compared in one trace.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def fused_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    values = (x.float() + y.float()) * 0.5
    return torch.nn.functional.silu(values).to(x.dtype)


def summarize_trace(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        events = json.load(source)["traceEvents"]

    cpu_kernels = [
        event
        for event in events
        if event.get("cat") == "cpu_op"
        and str(event.get("name", "")).startswith("triton_poi_")
    ]
    gpu_kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and str(event.get("name", "")).startswith("triton_poi_")
    ]
    names = sorted({event["name"] for event in gpu_kernels})
    counts = Counter(
        (
            event["name"],
            "has_external_id"
            if "External id" in event.get("args", {})
            else "missing_external_id",
        )
        for event in gpu_kernels
    )
    status_by_name = {
        name: {
            "has_external_id": counts[name, "has_external_id"],
            "missing_external_id": counts[name, "missing_external_id"],
        }
        for name in names
    }
    return {
        "trace": str(path),
        "torch_version": torch.__version__,
        "gpu_kernel_names": names,
        "gpu_status_by_kernel_name": status_by_name,
        "cpu_triton_events": len(cpu_kernels),
        "gpu_triton_events": len(gpu_kernels),
        "gpu_events_with_external_id": sum(
            counts[name, "has_external_id"] for name in names
        ),
        "gpu_events_missing_external_id": sum(
            counts[name, "missing_external_id"] for name in names
        ),
        "cpu_event_example": (
            {
                "name": cpu_kernels[0]["name"],
                "input_shapes": cpu_kernels[0].get("args", {}).get("Input Dims"),
                "input_dtypes": cpu_kernels[0].get("args", {}).get("Input type"),
                "kernel_file": cpu_kernels[0].get("args", {}).get("kernel_file"),
            }
            if cpu_kernels
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support BF16; use --dtype float32")

    trace_dir = args.output_dir / "profile"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "rank0.repro.pt.trace.json"
    if trace_path.exists():
        parser.error(f"trace already exists: {trace_path}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    x = torch.randn((512, 1024), device="cuda", dtype=dtype)
    y = torch.randn_like(x)
    compiled = torch.compile(fused_step, fullgraph=True)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            compiled(x, y)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = compiled(x, y)
    graph.replay()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        with record_function("direct_compiled_call"):
            direct_output = compiled(x, y)
        with record_function("cuda_graph_replays"):
            for _ in range(args.replays):
                graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    assert direct_output.shape == graph_output.shape

    summary = summarize_trace(trace_path)
    print(json.dumps(summary, indent=2))
    if not summary["gpu_kernel_names"]:
        raise SystemExit(
            "No triton_poi_ kernel was emitted; inspect the trace and compiler backend"
        )
    if not any(
        counts["has_external_id"] and counts["missing_external_id"]
        for counts in summary["gpu_status_by_kernel_name"].values()
    ):
        raise SystemExit(
            "No Triton kernel has both attributed and unattributed GPU events"
        )


if __name__ == "__main__":
    main()

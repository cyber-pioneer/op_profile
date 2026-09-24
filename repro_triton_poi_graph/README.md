# Triton pointwise kernel attribution reproducer

This standalone case shows why a fused `triton_poi_fused__*` kernel may have a
synthetic operator name and why CUDA Graph replay events may lack input shape,
dtype, and `External id` in a runtime profiler trace. It does not run Qwen or
identify the source APIs of Qwen's specific fused kernels.

Requirements: a CUDA-enabled PyTorch build with working `torch.compile` and
Triton. The case was verified with PyTorch 2.11.0+cu129 and Triton 3.6.0. It
does not require vLLM or vllm-plugin-FL.

From the `op_profile` repository root:

```bash
RUN_DIR=$(mktemp -d /tmp/triton_poi_repro.XXXXXX)
python3 repro_triton_poi_graph/repro_triton_poi_graph.py --output-dir "$RUN_DIR"
```

The script compiles conversion, addition, multiplication, and SiLU into an
Inductor kernel. It profiles one direct compiled call and eight CUDA Graph
replays, then prints event counts and a CPU event's input shape, dtype, and
generated `kernel_file`. The full trace is saved to
`$RUN_DIR/profile/rank0.repro.pt.trace.json`. Use `--dtype float32` if the GPU
does not support BF16, or `--replays N` to vary the replay count. Use a new run
directory for each attempt.

In the verified environment, the kernel was
`triton_poi_fused__to_copy_add_mul_silu_0`: one GPU event had `External id`,
and eight replay events did not. Exact generated names vary by PyTorch version
and backend. The original ATen calls have been fused, so the runtime kernel
name is not a one-to-one API name. Inspect the printed `kernel_file` and the
compiler graph when investigating a particular fusion.

If the vllm-plugin-FL operator-profile extractor is also available, it can
convert this trace to the same CSV format used for the Qwen runs:

```bash
python3 /path/to/vllm-plugin-FL/tools/operator_profile/extract_operator_shapes.py \
  --runtime "$RUN_DIR/profile" --rank 0 --output-dir "$RUN_DIR/results"
```

`kernel_shape_dtype.csv` will then show both `operator_shape_matched` and
`missing_external_id` for the same kernel; `operator_list.csv` uses the
generated Triton name rather than an original ATen API.

import copy
import json
import logging
from pathlib import Path

import boto3
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import helper, numpy_helper

from training.config import ModelConfig

logger = logging.getLogger(__name__)

QUANTIZATION_SCHEME = "onnx_weight_only_int8_per_channel"
OPSET_VERSION = 17

INPUT_NAMES = ["deck_a", "deck_b", "deck_a_lvls", "deck_b_lvls"]
OUTPUT_NAME = "logit"


class _ExportableRMSNorm(nn.Module):
    """Plain-tensor equivalent of ``nn.RMSNorm``.

    ``aten::rms_norm`` has no symbolic for the TorchScript ONNX exporter, so we
    swap the fused module for this decomposition just before export.
    """

    def __init__(self, src: nn.RMSNorm) -> None:
        super().__init__()
        self.weight = src.weight
        self.eps = src.eps if src.eps is not None else torch.finfo(torch.float32).eps
        self.dims = tuple(range(-len(src.normalized_shape), 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=self.dims, keepdim=True) + self.eps)
        return x if self.weight is None else x * self.weight


def _swap_rmsnorm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, _ExportableRMSNorm(child))
        else:
            _swap_rmsnorm(child)


def _example_inputs(model_cfg: ModelConfig) -> tuple[torch.Tensor, ...]:
    """A single-game batch matching ``TransformerBinaryClassifier.forward``."""
    b, s = 1, model_cfg.set_size
    cards = torch.zeros(b, s, dtype=torch.int64)
    lvls = torch.zeros(b, s, dtype=torch.float32)
    return cards, cards.clone(), lvls, lvls.clone()


def _gemm_to_matmul(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite every constant-weight ``Gemm`` (what ``nn.Linear`` exports to on 2D
    input) into ``MatMul`` (+ ``Add``), baking ``transB``/``alpha``/``beta`` into
    fresh initializers.

    This normalizes all Linear layers to a single ``MatMul`` form so
    ``_int8_quantize_matmul_weights`` picks them all up (the 2D-input Linears in
    the classification head would otherwise stay as ``Gemm`` and escape it).
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    rewritten = []
    for node in graph.node:
        if node.op_type != "Gemm":
            rewritten.append(node)
            continue

        attrs = {a.name: a for a in node.attribute}
        alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
        beta = attrs["beta"].f if "beta" in attrs else 1.0
        trans_a = attrs["transA"].i if "transA" in attrs else 0
        trans_b = attrs["transB"].i if "transB" in attrs else 0
        a_in, b_in = node.input[0], node.input[1]
        c_in = node.input[2] if len(node.input) > 2 else None

        if trans_a or b_in not in inits or (c_in is not None and c_in not in inits):
            rewritten.append(node)  # not a plain constant-weight Linear; leave as-is
            continue

        weight = numpy_helper.to_array(inits[b_in])
        if trans_b:
            weight = weight.T
        if alpha != 1.0:
            weight = weight * alpha
        mm_weight = f"{node.name}_w"
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(weight), mm_weight))

        mm_out = node.output[0] if c_in is None else f"{node.name}_mm"
        rewritten.append(helper.make_node("MatMul", [a_in, mm_weight], [mm_out], f"{node.name}_MatMul"))
        if c_in is not None:
            bias = c_in
            if beta != 1.0:
                bias = f"{node.name}_b"
                scaled = numpy_helper.to_array(inits[c_in]) * beta
                graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(scaled), bias))
            rewritten.append(helper.make_node("Add", [mm_out, bias], [node.output[0]], f"{node.name}_Add"))

    del graph.node[:]
    graph.node.extend(rewritten)
    return model


def _prune_dead(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop nodes whose outputs feed nothing and initializers nobody reads."""
    graph = model.graph
    outputs = {o.name for o in graph.output}
    while True:
        consumed = outputs | {name for node in graph.node for name in node.input}
        live = [n for n in graph.node if any(o in consumed for o in n.output)]
        if len(live) == len(graph.node):
            break
        del graph.node[:]
        graph.node.extend(live)

    referenced = {name for node in graph.node for name in node.input}
    kept = [i for i in graph.initializer if i.name in referenced]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return model


def _int8_quantize_matmul_weights(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Per-output-channel symmetric INT8 weight-only quantization of every
    constant-weight ``MatMul`` (i.e. the ``nn.Linear`` layers).

    Each float weight initializer is replaced by an INT8 tensor plus a
    per-channel scale, with a ``DequantizeLinear`` in front of the MatMul.
    onnxruntime folds that back to float at load, so this matches the previous
    scheme: INT8 on disk, no calibration data, no accuracy cliff. Embedding
    tables (read by ``Gather``) are left in FP32 - they are a rounding error of
    the file size and quantizing them measurably hurts accuracy.
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}

    identity_src = {n.output[0]: n.input[0] for n in graph.node if n.op_type == "Identity"}

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in identity_src and name not in seen:
            seen.add(name)
            name = identity_src[name]
        return name

    consumers: dict[str, list] = {}
    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            src = resolve(node.input[1])
            if src in inits and numpy_helper.to_array(inits[src]).ndim == 2:
                consumers.setdefault(src, []).append(node)

    dequant_nodes = []
    new_inits = []
    n_params = 0
    for wname, matmuls in consumers.items():
        weight = numpy_helper.to_array(inits[wname]).astype(np.float32)  # (in, out)
        scale = np.abs(weight).max(axis=0)
        scale = np.where(scale < 1e-8, 1e-8, scale).astype(np.float32) / 127.0
        q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)

        q_name, s_name, dq_name = f"{wname}_i8", f"{wname}_scale", f"{wname}_dq"
        new_inits.append(numpy_helper.from_array(q, q_name))
        new_inits.append(numpy_helper.from_array(scale, s_name))
        dequant_nodes.append(
            helper.make_node("DequantizeLinear", [q_name, s_name], [dq_name], dq_name, axis=1)
        )
        for matmul in matmuls:
            matmul.input[1] = dq_name
        n_params += weight.size

    kept = [i for i in graph.initializer if i.name not in consumers]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_inits)

    # DequantizeLinear only depends on initializers, so it is safe at the front.
    ordered = dequant_nodes + list(graph.node)
    del graph.node[:]
    graph.node.extend(ordered)

    return _prune_dead(model), n_params


def export_onnx(
    model: nn.Module,
    model_cfg: ModelConfig,
    vocab_size: int,
    local_path: Path,
) -> Path:
    """Export the trained model to an FP32 ONNX graph with a dynamic batch dim."""
    model = copy.deepcopy(model).eval().to("cpu")
    _swap_rmsnorm(model)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES}
    dynamic_axes[OUTPUT_NAME] = {0: "batch"}

    torch.onnx.export(
        model,
        _example_inputs(model_cfg),
        str(local_path),
        input_names=INPUT_NAMES,
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )

    # Stash the rebuild metadata on the graph so the model file stays self-describing.
    onnx_model = onnx.load(str(local_path))
    for key, value in {
        "model_config": json.dumps(model_cfg.model_dump()),
        "vocab_size": str(vocab_size),
    }.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(onnx_model, str(local_path))

    logger.info("exported FP32 ONNX graph to %s", local_path)
    return local_path


def quantize_onnx_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Rewrite the FP32 ONNX graph to store its ``nn.Linear`` weights as INT8."""
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    model = _gemm_to_matmul(onnx.load(str(fp32_path)))
    model, n_params = _int8_quantize_matmul_weights(model)
    scheme = model.metadata_props.add()
    scheme.key, scheme.value = "quantization", QUANTIZATION_SCHEME
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(int8_path))

    logger.info(
        "quantized %s Linear weight params to INT8; wrote %s (%.2f MB, was %.2f MB)",
        f"{n_params:,}",
        int8_path,
        int8_path.stat().st_size / 1e6,
        fp32_path.stat().st_size / 1e6,
    )
    return int8_path


def upload_model(local_path: Path, bucket: str, key: str) -> str:
    """Upload the model file to ``s3://bucket/key`` and return the S3 URI."""
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    logger.info("uploaded INT8 ONNX model to %s", s3_uri)
    return s3_uri

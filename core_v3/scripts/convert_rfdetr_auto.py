import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
import rfdetr.models.backbone.projector as _projector_mod
import rfdetr.models.ops.modules.ms_deform_attn as _ms_deform_mod


def _patch_export_compat():
    """Patch RF-DETR internals to make ONNX/TRT export path compatible."""

    def layernorm_forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (int(x.size(3)),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

    _projector_mod.LayerNorm.forward.__code__ = layernorm_forward.__code__

    def ms_deform_attn_forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes=None,
        input_level_start_index=None,
        input_padding_mask=None,
        input_spatial_shapes_hw=None,
        **kwargs,
    ):
        class MultiscaleDeformableAttnPlugin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
                value = value.permute(0, 2, 3, 1)
                n, lq, m, l, p, _ = sampling_locations.shape
                attention_weights = attention_weights.view(n, lq, m, l * p)
                # Import locally so injected code does not depend on outer-module globals.
                from rfdetr.models.ops.modules.ms_deform_attn import ms_deform_attn_core_pytorch
                return ms_deform_attn_core_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights
                )

            @staticmethod
            def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
                return g.op(
                    "TRT::MultiscaleDeformableAttnPlugin_TRT",
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                )

        if input_spatial_shapes is None:
            input_spatial_shapes = input_spatial_shapes_hw
        if input_spatial_shapes is None:
            raise ValueError("input_spatial_shapes is required")

        n, len_q, _ = query.shape
        _, len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(
            n, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n, len_q, self.n_heads, self.n_levels * self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(
            n, self.n_heads, self.d_model // self.n_heads, len_in
        )
        value = value.permute(0, 3, 1, 2)

        l, p = sampling_locations.shape[3:5]
        attention_weights = attention_weights.view(n, len_q, self.n_heads, l, p)

        output = MultiscaleDeformableAttnPlugin.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
        )
        output = output.view(n, len_q, self.d_model)
        return self.output_proj(output)

    _ms_deform_mod.MSDeformAttn.forward.__code__ = ms_deform_attn_forward.__code__


class DeepStreamOutput(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def forward(self, x):
        boxes = x[0]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes @= convert_matrix
        boxes *= torch.as_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        scores = x[1].sigmoid()
        scores, labels = torch.max(scores, dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


VARIANT_FACTORIES = {
    "rfdetr-nano": RFDETRNano,
    "rfdetr-small": RFDETRSmall,
    "rfdetr-medium": RFDETRMedium,
    "rfdetr-base": RFDETRBase,
    "rfdetr-large": RFDETRLarge,
}


def _normalize_class_map(class_names):
    if isinstance(class_names, dict):
        return class_names
    if isinstance(class_names, list):
        return {idx + 1: name for idx, name in enumerate(class_names)}
    return {}


def _write_labels(labels_path: Path, num_classes: int, class_map: dict):
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as f:
        f.write("background\n")
        for i in range(1, num_classes + 1):
            f.write(f"{class_map.get(i, 'empty')}\n")


def _load_first_compatible(variants, weights, resolution, device):
    errors = {}
    for variant in variants:
        factory = VARIANT_FACTORIES[variant]
        try:
            print(f"[TRY] {variant}")
            model = factory(pretrain_weights=weights, resolution=resolution, device=device.type)
            print(f"[OK]  Compatible variant: {variant}")
            return variant, model
        except Exception as exc:
            errors[variant] = str(exc)
            print(f"[SKIP] {variant}: {exc}")

    details = "\n".join([f"- {k}: {v}" for k, v in errors.items()])
    raise RuntimeError(f"No compatible RF-DETR variant found for checkpoint:\n{details}")


def _build_export_model(rfdetr_model, img_size, device):
    num_classes = rfdetr_model.model_config.num_classes
    class_map = _normalize_class_map(rfdetr_model.class_names)

    model = deepcopy(rfdetr_model.model.model)
    model.to(device)
    model.eval()
    if hasattr(model, "export"):
        model.export()

    export_model = nn.Sequential(model, DeepStreamOutput(img_size))
    export_model.eval()
    return export_model, num_classes, class_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-detect RF-DETR variant and convert .pth checkpoint to ONNX."
    )
    parser.add_argument("-w", "--weights", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "-s",
        "--size",
        nargs="+",
        type=int,
        default=[640],
        help="Inference size [H,W] (default: 640)",
    )
    parser.add_argument(
        "--variants",
        default="rfdetr-nano,rfdetr-small,rfdetr-medium,rfdetr-base,rfdetr-large",
        help="Comma-separated variant priority list",
    )
    parser.add_argument("--output", default=None, help="Output ONNX path (default: <weights>.onnx)")
    parser.add_argument("--labels-output", default=None, help="Output labels path (default: labels.txt beside ONNX)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=1, help="Static batch size")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument("--simplify", action="store_true", help="Run onnxslim simplification")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        raise RuntimeError(f"Invalid weights file: {args.weights}")
    if len(args.size) > 1 and args.size[0] != args.size[1]:
        raise RuntimeError("RF-DETR model requires square resolution")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set --dynamic together with --batch > 1")

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    resolution = img_size[0]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_FACTORIES:
            raise RuntimeError(f"Unknown variant '{v}'. Available: {list(VARIANT_FACTORIES.keys())}")

    weights_path = Path(args.weights)
    onnx_output = Path(args.output) if args.output else weights_path.with_suffix(".onnx")
    labels_output = Path(args.labels_output) if args.labels_output else onnx_output.parent / "labels.txt"

    _patch_export_compat()
    device = torch.device("cpu")

    print(f"[INFO] Weights: {weights_path}")
    print(f"[INFO] Output : {onnx_output}")
    print(f"[INFO] Labels : {labels_output}")
    print(f"[INFO] Size   : {img_size[0]}x{img_size[1]}")
    print(f"[INFO] Dynamic batch: {args.dynamic}")

    variant, loaded = _load_first_compatible(variants, str(weights_path), resolution, device)
    export_model, num_classes, class_map = _build_export_model(loaded, img_size, device)
    _write_labels(labels_output, num_classes, class_map)
    print(f"[INFO] Labels written: {labels_output}")

    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(args.batch, 3, *img_size, device=device)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    print(f"[INFO] Exporting ONNX using variant '{variant}'...")
    # Use legacy exporter path to avoid torch.export symbolic guard issues.
    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_output),
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    if args.simplify:
        print("[INFO] Simplifying ONNX...")
        import onnx
        import onnxslim

        model_onnx = onnx.load(str(onnx_output))
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, str(onnx_output))

    print(f"[DONE] ONNX exported: {onnx_output}")
    print(f"[DONE] Detected variant: {variant}")


if __name__ == "__main__":
    main()

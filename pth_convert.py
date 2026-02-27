import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

MODEL_PATH = "/app/raw_models/ai-model_20251027_040415_a45752f9-fa10-471f-87f5-a9858be906d6.pth"
ONNX_OUTPUT = "rfdetr_raw.onnx"
IMG_SIZE = 576  # must match training resolution


# -------------------------------------------------
# 1️⃣ Disable antialias (important for ONNX)
# -------------------------------------------------

_original_interpolate = F.interpolate

def patched_interpolate(*args, **kwargs):
    kwargs["antialias"] = False
    return _original_interpolate(*args, **kwargs)

F.interpolate = patched_interpolate


# -------------------------------------------------
# 2️⃣ Load RF-DETR variant
# -------------------------------------------------

variants = [
    ("Nano", RFDETRNano),
    ("Small", RFDETRSmall),
    ("Medium", RFDETRMedium),
    ("Base", RFDETRBase),
    ("Large", RFDETRLarge)
]

rfdetr = None

for name, cls in variants:
    try:
        print(f"🔍 Trying {name}")
        rfdetr = cls(pretrain_weights=MODEL_PATH)
        print(f"✅ Loaded {name}")
        break
    except Exception as e:
        print(f"❌ {name} failed: {e}")

if rfdetr is None:
    raise RuntimeError("Could not load RF-DETR variant")


# -------------------------------------------------
# 3️⃣ Extract real nn.Module
# -------------------------------------------------

model = rfdetr.model.model
model = model.cpu()
model.eval()

print("✅ Model ready")


# -------------------------------------------------
# 4️⃣ Wrap to return raw outputs only
# -------------------------------------------------

class RFDETRRawWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["pred_logits"], outputs["pred_boxes"]


export_model = RFDETRRawWrapper(model)
export_model.eval()


# -------------------------------------------------
# 5️⃣ Dummy input
# -------------------------------------------------

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)


# -------------------------------------------------
# 6️⃣ Export using Dynamo exporter (CRITICAL)
# -------------------------------------------------

torch.onnx.export(
    export_model,
    dummy,
    ONNX_OUTPUT,
    opset_version=18,
    input_names=["input"],
    output_names=["pred_logits", "pred_boxes"],
    dynamic_axes={
        "input": {0: "batch"},
        "pred_logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    },
    dynamo=True  # 🔥 THIS IS THE IMPORTANT PART
)

print(f"🚀 Exported successfully → {ONNX_OUTPUT}")

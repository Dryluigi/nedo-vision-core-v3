import onnx

model = onnx.load("ai-model_20251027_040415_a45752f9-fa10-471f-87f5-a9858be906d6.onnx")
# model = onnx.load("yolo11s_dynamic.onnx")

print("=== INPUT ===")
for inp in model.graph.input:
    name = inp.name
    shape = [
        dim.dim_value if dim.dim_value > 0 else "dynamic"
        for dim in inp.type.tensor_type.shape.dim
    ]
    print(name, shape)

print("\n=== OUTPUT ===")
for out in model.graph.output:
    name = out.name
    shape = [
        dim.dim_value if dim.dim_value > 0 else "dynamic"
        for dim in out.type.tensor_type.shape.dim
    ]
    print(name, shape)

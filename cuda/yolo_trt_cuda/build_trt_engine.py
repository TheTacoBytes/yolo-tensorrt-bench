import tensorrt as trt

ONNX_MODEL = "yolo11n-seg.onnx"
TRT_ENGINE  = "yolo11n.trt"

logger  = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser  = trt.OnnxParser(network, logger)

# 1) Parse ONNX
with open(ONNX_MODEL, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

# 2) Build config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# config.set_flag(trt.BuilderFlag.FP16)  # optionally enable FP16

# 3) Build serialized engine
print("Building serialized TensorRT network… (this may take a minute)")
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Network serialization failed")

# 4) Write engine file
with open(TRT_ENGINE, "wb") as f:
    f.write(serialized_engine)
print(f"✅ Engine saved as {TRT_ENGINE}")

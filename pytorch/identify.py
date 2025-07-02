import tensorrt as trt

logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flags   = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser  = trt.OnnxParser(network, logger)

# Parse your ONNX
with open("yolo11n-seg.onnx","rb") as f:
    parser.parse(f.read())

# Open an output file
with open("./pytorch/layer_info.txt", "w") as out_f:
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            t = layer.get_output(j)
            line = f"layer {i:02d}  {layer.type.name:12s} → tensor `{t.name}`  shape={tuple(t.shape)}\n"
            out_f.write(line)

print("Wrote layer info to layer_info.txt")

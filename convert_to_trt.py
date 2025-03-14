import tensorrt as trt

onnx_path = "/kaggle/input/YOUR_DATASET_NAME/model.onnx"
trt_path = "/kaggle/working/model.trt"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        print("Failed to parse ONNX model!")
        exit(1)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

with open(trt_path, "wb") as f:
    f.write(engine.serialize())

print(f"TensorRT model saved at {trt_path}")

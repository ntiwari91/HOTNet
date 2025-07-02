import torch
from fvcore.nn import FlopCountAnalysis
from model_original import S3RNet

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (consistent with training script)
in_channels = 3
out_channels = 31
height = 64
width = 64
scale = 4
alpha = 0.2

# Initialize model and move to device
model = S3RNet(in_channels=in_channels, out_channels=out_channels, ratio=scale).to(device)
model.eval()

# Dummy input on same device
dummy_input = torch.randn(1, in_channels, height, width).to(device)

# Get only the first output for FLOPs estimation (if model returns multiple outputs)
with torch.no_grad():
    output = model(dummy_input)
    if isinstance(output, (list, tuple)):
        output = output[0]

# Compute FLOPs
flops = FlopCountAnalysis(model, dummy_input)
print(f"Total FLOPs: {flops.total() / 1e9:.4f} GFLOPs")

import torch
import sys
sys.path.insert(0, '.')  # so it finds RRDBNet_arch.py
import RRDBNet_arch as arch

# ── CONFIG ──────────────────────────────────────────────
WEIGHTS_PATH = 'models/RRDB_ESRGAN_x4.pth'  # change if your .pth is elsewhere
OUTPUT_PATH  = 'esrgan.onnx'
# ────────────────────────────────────────────────────────

# Load YOUR custom architecture (nf=128, nb=32, gc=64)
model = arch.RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)

# Load weights
checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Model loaded successfully")

# Dummy input — small patch, dynamic axes handle real sizes
dummy = torch.rand(1, 3, 64, 64)

# Export
torch.onnx.export(
    model,
    dummy,
    OUTPUT_PATH,
    opset_version=11,
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'}
    }
)

print(f"✅ Exported to {OUTPUT_PATH}")

# Quick verify
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession(OUTPUT_PATH)
test = np.random.rand(1, 3, 64, 64).astype(np.float32)
out  = sess.run(None, {'input': test})
print(f"✅ Verified — output shape: {out[0].shape}")
print(f"✅ File size: {__import__('os').path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")
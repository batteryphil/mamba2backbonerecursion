import os
import sys
import torch

# Explicitly add CUDA and Torch DLL directories for Windows
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_path):
    print(f"Adding CUDA DLL directory: {cuda_path}")
    os.add_dll_directory(cuda_path)

# Add torch lib directory
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    print(f"Adding Torch DLL directory: {torch_lib_path}")
    os.add_dll_directory(torch_lib_path)

try:
    import mamba_scan
    print("Import SUCCESSFUL!")
    x = torch.randn(2, 64, 128).cuda()
    dt = torch.randn(2, 64, 128).cuda()
    A = torch.randn(128, 16).cuda()
    B = torch.randn(2, 64, 16).cuda()
    C = torch.randn(2, 64, 16).cuda()
    D = torch.randn(128).cuda()
    y = mamba_scan.ssm_scan_fwd(x, dt, A, B, C, D)
    print("Kernel execution SUCCESSFUL! Output shape:", y.shape)
except Exception as e:
    print(f"Import or Execution FAILED: {e}")
    import traceback
    traceback.print_exc()

import torch, os
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("allow_tf32 (legacy getter) try:")
try:
    print(torch.backends.cuda.matmul.allow_tf32)
except Exception as e:
    print("ERROR reading allow_tf32:", repr(e))
print("env TORCH_ALLOW_TF32_CUBLAS_OVERRIDE =", os.getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))
print("env NVIDIA_TF32_OVERRIDE =", os.getenv("NVIDIA_TF32_OVERRIDE"))
print("matmul precision API exists:", hasattr(torch, "set_float32_matmul_precision"))

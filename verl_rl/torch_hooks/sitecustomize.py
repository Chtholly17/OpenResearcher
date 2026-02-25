import os

# 让这个 hook 可控：默认开启；如果你想临时跳过，设置 DISABLE_TORCH_HOOKS=1
if os.getenv("DISABLE_TORCH_HOOKS") == "1":
    raise SystemExit

import torch

# ---------------------------
# 关键：只用“新 API”控制 TF32/FP32 precision
# ---------------------------
# 可选值通常是: "tf32", "ieee"
# - tf32：性能更好（推荐你推理用）
# - ieee：更严格的 FP32（更稳定/一致性更好但可能慢）
PREC = os.getenv("TORCH_FP32_PRECISION", "tf32")

# 新 API：matmul / conv / rnn 的 fp32 精度策略
torch.backends.cuda.matmul.fp32_precision = PREC
torch.backends.cudnn.conv.fp32_precision = PREC
torch.backends.cudnn.rnn.fp32_precision = PREC

# 可选：如果你希望完全避免 torch.set_float32_matmul_precision 被别处改写，可以强制一次
# 注意：这个 API 的有效取值是 "highest"/"high"/"medium"（和上面的 tf32/ieee不是一套枚举）
# 这里不强制设置，避免引入另一套策略；你若要用也行，但要统一口径。

# ---------------------------
# 调试输出（可选）
# ---------------------------
if os.getenv("TORCH_HOOKS_VERBOSE") == "1":
    try:
        print("[sitecustomize] torch:", torch.__version__)
        print("[sitecustomize] matmul.fp32_precision =", torch.backends.cuda.matmul.fp32_precision)
        print("[sitecustomize] cudnn.conv.fp32_precision =", torch.backends.cudnn.conv.fp32_precision)
        # legacy getter 只读不写（读可能会 warning，但不应再 error）
        print("[sitecustomize] legacy allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
    except Exception as e:
        print("[sitecustomize] error while printing torch settings:", repr(e))
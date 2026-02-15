PYTHONPATH="$PROJECT_ROOT/torch_hooks:$PYTHONPATH" \
TORCH_FP32_PRECISION=tf32 \
TORCH_HOOKS_VERBOSE=1 \
python - <<'PY'
import torch
print("matmul.fp32_precision:", torch.backends.cuda.matmul.fp32_precision)
print("legacy allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
PY

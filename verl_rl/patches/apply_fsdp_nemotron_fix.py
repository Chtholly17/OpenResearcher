#!/usr/bin/env python3
"""
Apply the configuration_nemotron_h import fix to verl's fsdp_workers.py.

Nemotron models use `from configuration_nemotron_h import ...` (absolute import).
The configuration_* module lives in the model repo, not as a pip package.
This patch adds the model directory to sys.path before loading so the import succeeds.

Run from OpenResearcher root: python verl_rl/patches/apply_fsdp_nemotron_fix.py
Re-apply after `pip install --upgrade verl` or reinstalling verl.
"""

FSDP_PATCH = '''
        # Workaround for models with custom code that use absolute imports (e.g. Nemotron:
        # "from configuration_nemotron_h import ..."). The configuration_* modules live in the
        # model repo, not as pip packages. Add the model dir to sys.path so they can be imported.
        import sys

        def _ensure_model_dir_in_path(path: str) -> None:
            if not path:
                return
            model_dir = path
            if not os.path.isdir(path):
                # HuggingFace model ID - resolve to cache dir by fetching a small file
                if "/" in path and not path.startswith("/") and not path.startswith("hdfs:"):
                    try:
                        from huggingface_hub import hf_hub_download

                        # Try config file first (Nemotron), fallback to config.json for dir
                        for fname in ("configuration_nemotron_h.py", "config.json"):
                            try:
                                fpath = hf_hub_download(repo_id=path, filename=fname)
                                model_dir = os.path.dirname(fpath)
                                break
                            except Exception:
                                continue
                        else:
                            return
                    except Exception:
                        return
                else:
                    return
            if model_dir and model_dir not in sys.path:
                sys.path.insert(0, model_dir)

        _ensure_model_dir_in_path(local_path)

'''

# Insert patch between local_path = model_path and the tokenizer line
MARKER_OLD = (
    "        local_path = model_path\n\n"
    "        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect\n"
    "        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly\n"
    "        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)"
)
MARKER_NEW = (
    "        local_path = model_path\n" + FSDP_PATCH +
    "        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect\n"
    "        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly\n"
    "        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)"
)
PATCH_ALREADY_APPLIED = "_ensure_model_dir_in_path"


def main():
    import verl.workers.fsdp_workers as mod

    path = mod.__file__
    with open(path, "r") as f:
        content = f.read()

    if PATCH_ALREADY_APPLIED in content:
        print(f"Patch already applied to {path}")
        return 0

    if MARKER_OLD not in content:
        print(f"ERROR: Could not find insertion point in {path}")
        print("verl version may have changed. Please apply the patch manually.")
        return 1

    new_content = content.replace(MARKER_OLD, MARKER_NEW)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"Applied configuration_nemotron_h fix to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

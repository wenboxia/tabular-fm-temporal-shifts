"""设备解析（Phase 5.5）。

- 实测 TabPFN 在 Apple GPU (MPS) 上比 CPU 慢 5.6×，所以**永远不选 mps**。
- "auto" = 有 NVIDIA 显卡就用 cuda，否则 cpu。

**GPU 机器上的硬约束**：设置环境变量 ``NEURAL1_REQUIRE_CUDA=1`` 后，任何会落到 CPU 的
解析都直接抛错，而不是悄悄退回 CPU。正式实验在 CPU 上跑太慢，没有意义；与其跑几个小时才发现在用 CPU，不如第一秒就停下。环境变量会被
``run_multiseed.py`` 启动的每个子进程继承，因此不需要把参数层层传递。
"""
import os

import torch

ALLOWED = ("auto", "cpu", "cuda")
REQUIRE_ENV = "NEURAL1_REQUIRE_CUDA"


def cuda_required() -> bool:
    return os.environ.get(REQUIRE_ENV, "").strip() == "1"


def resolve_device(pref: str = "auto") -> str:
    if pref not in ALLOWED:
        raise ValueError(f"device 必须 ∈ {ALLOWED}，收到 {pref!r}（不支持 mps，实测比 cpu 慢）")

    have_cuda = torch.cuda.is_available()
    if cuda_required():
        if pref == "cpu":
            raise RuntimeError(
                f"{REQUIRE_ENV}=1 但请求了 device='cpu'。本机要求实验必须在 GPU 上跑。"
            )
        if not have_cuda:
            raise RuntimeError(
                f"{REQUIRE_ENV}=1 但 torch.cuda.is_available() 为 False —— 拒绝退回 CPU。"
                f" torch={torch.__version__} torch.version.cuda={torch.version.cuda}。"
                " 可以用 python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\" 检查。"
            )
        return "cuda"

    if pref == "auto":
        return "cuda" if have_cuda else "cpu"
    if pref == "cuda" and not have_cuda:
        raise RuntimeError("要求 cuda 但 torch.cuda.is_available() 为 False：驱动或 CUDA 版 torch 未装好")
    return pref


def describe(device: str) -> str:
    """一行可读的设备描述，写进日志作为"确实在 GPU 上跑"的证据。"""
    if device == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"cuda ({p.name}, {p.total_memory / 2**30:.1f} GB, torch {torch.__version__}, CUDA {torch.version.cuda})"
    return f"{device} (torch {torch.__version__})"

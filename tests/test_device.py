"""设备解析：GPU 机器上必须用 GPU，不许悄悄退回 CPU。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch
from src.utils import device as dev


def test_auto_without_requirement_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv(dev.REQUIRE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.resolve_device("auto") == "cpu"


def test_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.delenv(dev.REQUIRE_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert dev.resolve_device("auto") == "cuda"


def test_required_cuda_refuses_silent_cpu_fallback(monkeypatch):
    monkeypatch.setenv(dev.REQUIRE_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="拒绝退回 CPU"):
        dev.resolve_device("auto")


def test_required_cuda_refuses_explicit_cpu(monkeypatch):
    monkeypatch.setenv(dev.REQUIRE_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="必须在 GPU"):
        dev.resolve_device("cpu")


def test_required_cuda_resolves_to_cuda(monkeypatch):
    monkeypatch.setenv(dev.REQUIRE_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert dev.resolve_device("auto") == "cuda"


def test_mps_is_rejected():
    with pytest.raises(ValueError, match="mps"):
        dev.resolve_device("mps")

"""
Phase 4 A 单元测试：AdapterLibrary

不依赖 TabPFN，纯 torch + numpy，离线快速运行。
覆盖：
  1. 初始化：默认含 1 个 adapter，active_id == 0
  2. forward 形状正确
  3. 拟合差时 route 创建新 adapter
  4. 拟合好时 route 复用现有 adapter（不新建）
  5. 达到 max_adapters 后强制复用最佳现有，不再新建
  6. 非 active adapter 不接梯度（active_optimizer.step 后只有 active 的参数变化）
  7. drop-in 替换 GatedEnsemble.adapter（接口兼容）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.regime.adapter_library import AdapterLibrary
from src.models.gated_ensemble import GatedEnsemble


class TestAdapterLibraryBasics:

    def test_init_creates_one_adapter(self):
        torch.manual_seed(0)
        lib = AdapterLibrary(input_dim=10)
        assert lib.n_adapters() == 1
        assert lib.active_id == 0

    def test_forward_shape(self):
        torch.manual_seed(0)
        lib = AdapterLibrary(input_dim=10, hidden_dim=32, n_outputs=1)
        x = torch.randn(8, 10)
        y = lib(x)
        assert y.shape == (8, 1), f"forward 输出 shape 应 (8,1)，实为 {y.shape}"


class TestAdapterLibraryRouting:

    def test_route_creates_new_when_existing_poor_fit(self):
        """当现有 adapter 拟合 loss > fit_threshold 时应新建。"""
        torch.manual_seed(1)
        rng = np.random.default_rng(1)
        lib = AdapterLibrary(
            input_dim=5, hidden_dim=16, fit_threshold=0.01, max_adapters=4,
        )
        # 注入完全无规律的目标，初始随机 adapter 几乎肯定 loss > 0.01
        X = rng.normal(size=(50, 5)).astype(np.float32)
        # 目标 e ∈ [-0.5, 0.5] 比较"乱"
        e = rng.uniform(-0.5, 0.5, size=50).astype(np.float32)

        prev_n = lib.n_adapters()
        active_id, action, losses = lib.route(X, e, t=100)
        assert action == "create", f"应新建 adapter，实际 action={action}, losses={losses}"
        assert lib.n_adapters() == prev_n + 1
        assert active_id == 1, f"新 adapter id 应为 1，实为 {active_id}"

    def test_route_reuses_when_well_fitted(self):
        """先 train 一个 adapter 到低 loss，再次 route 应复用。"""
        torch.manual_seed(2)
        rng = np.random.default_rng(2)
        lib = AdapterLibrary(
            input_dim=5, hidden_dim=16, fit_threshold=0.05, max_adapters=4, lr=1e-2,
        )

        # 制造可学习的线性目标 e = sum(X)/5
        X = rng.normal(size=(80, 5)).astype(np.float32)
        e = X.sum(axis=1) / 5.0

        # train 当前 active adapter 拟合 (X, e)
        X_t = torch.tensor(X)
        e_t = torch.tensor(e)
        opt = lib.active_optimizer()
        for _ in range(300):
            pred = lib(X_t).view(-1)
            loss = torch.nn.functional.mse_loss(pred, e_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 现在 active 已拟合好；再 route 应判定为复用（reuse）
        active_id, action, losses = lib.route(X, e, t=200)
        assert action == "reuse", (
            f"应复用现有 active adapter，实际 action={action}, losses={losses}"
        )
        assert active_id == 0
        assert lib.n_adapters() == 1, "不应新建 adapter"

    def test_max_adapters_cap_forces_reuse(self):
        """达到 max_adapters 后，即使 loss 高也强制复用最佳现有。"""
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        lib = AdapterLibrary(
            input_dim=5, hidden_dim=16, fit_threshold=1e-9, max_adapters=2,
        )
        X = rng.normal(size=(60, 5)).astype(np.float32)
        e = rng.uniform(-1, 1, size=60).astype(np.float32)

        # 第 1 次 route：应新建（创建 adapter 1，库内 [0,1]）
        _, action1, _ = lib.route(X, e, t=100)
        assert action1 == "create"
        assert lib.n_adapters() == 2

        # 第 2 次 route：fit_threshold 极小不可能达到 + 已达 max → 应 switch/reuse 而非 create
        _, action2, _ = lib.route(X, e, t=200)
        assert action2 in ("switch", "reuse"), (
            f"达到 max_adapters 后不应再 create，实际 action={action2}"
        )
        assert lib.n_adapters() == 2

    def test_route_history_records_events(self):
        torch.manual_seed(4)
        rng = np.random.default_rng(4)
        lib = AdapterLibrary(
            input_dim=4, hidden_dim=8, fit_threshold=1e-6, max_adapters=4,
        )
        X = rng.normal(size=(40, 4)).astype(np.float32)
        e = rng.normal(size=40).astype(np.float32)
        for t in [100, 250, 400]:
            lib.route(X, e, t=t)
        assert [r[0] for r in lib.route_history] == [100, 250, 400]
        assert all(isinstance(r[1], int) for r in lib.route_history)


class TestAdapterLibraryIsolation:

    def test_only_active_adapter_receives_gradient(self):
        """active_optimizer.step 后，只有 active adapter 的参数变化；其他冻结。"""
        torch.manual_seed(5)
        rng = np.random.default_rng(5)
        lib = AdapterLibrary(
            input_dim=4, hidden_dim=8, fit_threshold=1e-9, max_adapters=4, lr=1e-2,
        )
        # 触发新建 adapter 1（因为 fit_threshold 极小）
        X_np = rng.normal(size=(30, 4)).astype(np.float32)
        e_np = rng.normal(size=30).astype(np.float32)
        lib.route(X_np, e_np, t=100)
        assert lib.active_id == 1

        # 快照 adapter 0 第一线性层权重
        w0_before = lib.adapters["0"][0].weight.detach().clone()
        w1_before = lib.adapters["1"][0].weight.detach().clone()

        # 用 active（=1）做一次 backward + step
        X_t = torch.tensor(X_np)
        e_t = torch.tensor(e_np)
        opt = lib.active_optimizer()
        pred = lib(X_t).view(-1)
        loss = torch.nn.functional.mse_loss(pred, e_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        w0_after = lib.adapters["0"][0].weight.detach()
        w1_after = lib.adapters["1"][0].weight.detach()

        assert torch.allclose(w0_before, w0_after), (
            "非 active adapter 0 的参数不应被修改"
        )
        assert not torch.allclose(w1_before, w1_after), (
            "active adapter 1 的参数应被更新"
        )


class TestAdapterLibraryWarmStart:

    def test_first_adapter_is_random_init(self):
        """init 时创建的 adapter 0 走随机初始化分支。"""
        torch.manual_seed(20)
        lib = AdapterLibrary(input_dim=4, hidden_dim=8)
        assert lib.n_random_inits == 1
        assert lib.n_warmstart_inits == 0

    def test_subsequent_creates_warm_start_from_active(self):
        """create-driven 新 adapter 应从当前 active 复制权重，不再随机。"""
        torch.manual_seed(21)
        rng = np.random.default_rng(21)
        lib = AdapterLibrary(
            input_dim=4, hidden_dim=8, fit_threshold=1e-9, max_adapters=4, lr=1e-2,
        )
        # 把 adapter 0 训练到一个非随机状态
        X = rng.normal(size=(40, 4)).astype(np.float32)
        e = rng.normal(size=40).astype(np.float32)
        X_t = torch.tensor(X)
        e_t = torch.tensor(e)
        opt = lib.active_optimizer()
        for _ in range(50):
            pred = lib(X_t).view(-1)
            loss = torch.nn.functional.mse_loss(pred, e_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        w0 = lib.adapters["0"][0].weight.detach().clone()

        # 触发 routing 新建 → 应 warm-start 自 adapter 0
        lib.route(X, e, t=100)
        assert lib.active_id == 1
        assert lib.n_warmstart_inits == 1, (
            f"第二个 adapter 应 warm-start，n_warmstart_inits={lib.n_warmstart_inits}"
        )
        assert lib.n_random_inits == 1, (
            f"random_inits 不应增加，仍应是 1（adapter 0 那次）"
        )

        w1_initial = lib.adapters["1"][0].weight.detach()
        assert torch.allclose(w0, w1_initial), (
            "warm-start 后 adapter 1 的初始权重应与 adapter 0 完全一致"
        )

    def test_random_init_strategy_skips_warmstart(self):
        """init_strategy='random' 时，即使有 active 也不 warm-start，全部走随机分支。"""
        torch.manual_seed(30)
        rng = np.random.default_rng(30)
        lib = AdapterLibrary(
            input_dim=4, hidden_dim=8, fit_threshold=1e-9, max_adapters=4, lr=1e-2,
            init_strategy="random",
        )
        # adapter 0 训练成非随机
        X = rng.normal(size=(40, 4)).astype(np.float32)
        e = rng.normal(size=40).astype(np.float32)
        opt = lib.active_optimizer()
        for _ in range(50):
            pred = lib(torch.tensor(X)).view(-1)
            loss = torch.nn.functional.mse_loss(pred, torch.tensor(e))
            opt.zero_grad()
            loss.backward()
            opt.step()
        w0 = lib.adapters["0"][0].weight.detach().clone()

        # 触发 routing
        lib.route(X, e, t=100)
        assert lib.active_id == 1
        assert lib.n_warmstart_inits == 0, (
            f"init_strategy='random' 不应 warm-start，n_warmstart_inits={lib.n_warmstart_inits}"
        )
        assert lib.n_random_inits == 2, (
            f"应有 2 次 random_inits（adapter 0 + adapter 1），实际 {lib.n_random_inits}"
        )
        w1 = lib.adapters["1"][0].weight.detach()
        assert not torch.allclose(w0, w1), (
            "random init 下 adapter 1 不应与 adapter 0 完全一致（应是新随机）"
        )

    def test_warm_start_optimizer_state_is_fresh(self):
        """新 adapter 的 Adam 应是新建的，state 重置（不继承 active 的 momentum）。"""
        torch.manual_seed(22)
        rng = np.random.default_rng(22)
        lib = AdapterLibrary(
            input_dim=4, hidden_dim=8, fit_threshold=1e-9, max_adapters=4, lr=1e-2,
        )
        X = rng.normal(size=(20, 4)).astype(np.float32)
        e = rng.normal(size=20).astype(np.float32)
        # 训练 adapter 0 让它积累 Adam state
        opt0 = lib.active_optimizer()
        for _ in range(20):
            pred = lib(torch.tensor(X)).view(-1)
            loss = torch.nn.functional.mse_loss(pred, torch.tensor(e))
            opt0.zero_grad()
            loss.backward()
            opt0.step()
        # adapter 0 state 应非空
        state0 = list(opt0.state.values())
        assert len(state0) > 0
        assert "exp_avg" in state0[0]

        # 触发 routing
        lib.route(X, e, t=100)
        opt1 = lib.active_optimizer()
        # adapter 1 的 Adam state 应为空（还没 step 过）
        assert len(list(opt1.state.values())) == 0, (
            "warm-start 后新 adapter optimizer state 应为空（重置）"
        )


class TestAdapterLibraryDropIn:

    def test_drop_in_replacement_for_gated_ensemble_adapter(self):
        """把 GatedEnsemble.adapter 替换为 AdapterLibrary，forward 正常。"""
        torch.manual_seed(6)
        ge = GatedEnsemble(input_dim=10, hidden_dim=64, n_outputs=1)
        lib = AdapterLibrary(input_dim=10, hidden_dim=64, n_outputs=1)
        ge.adapter = lib  # drop-in

        x = torch.randn(4, 10)
        y_slow = torch.rand(4, 1)
        correction = torch.rand(4, 1)
        y_final, weights = ge(x, y_slow, correction)

        assert y_final.shape == (4, 1)
        assert weights.shape == (4, 3)

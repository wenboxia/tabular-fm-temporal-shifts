"""
Phase 4 A — Per-Regime Adapter Library

维护一组 per-regime MLP adapter，提供硬路由（hard routing）接口：
  - 当前 active adapter 接收 GatedEnsemble.forward 的调用与 consolidation 的梯度
  - 非 active adapter 完全冻结（不接梯度；其参数不出现在任何 active optimizer 中）
  - 检测到漂移（由外部 ADWINErrorDetector 触发）后，
    调用 route(X_recent, errors_recent) 决定：
      a) 切换到现有最适配的 adapter（若拟合 loss ≤ fit_threshold）
      b) 否则新建第 K+1 个 adapter（直至 max_adapters 上限）

设计取舍：
  - drop-in 替换 GatedEnsemble.adapter：forward(x) → (B, n_outputs)，
    与 nn.Sequential adapter 接口一致，便于不改 GatedEnsemble 直接接入
  - 每个 adapter 自带 Adam optimizer（lr 由初始化时指定），
    consolidation 调用 active_optimizer() 获取当前应更新的 optimizer
  - 路由时新建的 adapter 是空白随机初始化（per-regime 隔离的精神：
    每个 regime 独立从零学习，不继承上一 regime 的偏置）
  - 评估"现有 adapter 是否够用"用 MSE(adapter(X_recent), errors_recent)，
    与 FastToInterConsolidation 的训练目标一致 —— 拟合 fast corrector 的近期误差
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterLibrary(nn.Module):
    """
    Per-regime MLP adapter 字典 + 硬路由。

    drop-in 替换 GatedEnsemble.adapter 的 nn.Sequential：
      lib = AdapterLibrary(input_dim=10)
      gated_ensemble.adapter = lib  # 此后 gated_ensemble.adapter(x) 路由到 active
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_outputs: int = 1,
        max_adapters: int = 8,
        fit_threshold: float = 0.05,
        lr: float = 1e-3,
        init_strategy: str = "warm",
    ):
        """
        Args:
            input_dim:     原始特征维度
            hidden_dim:    每个 adapter 的隐藏层宽度（与 GatedEnsemble.adapter 对齐）
            n_outputs:     输出维度（二分类取 1）
            max_adapters:  字典上限。超过后 route() 强制复用现有最佳，不再新建
            fit_threshold: 评估现有 adapter 时的 MSE 上限。
                           min_loss ≤ threshold → 复用；否则若未到 max → 新建
            lr:            每个 adapter 自带 Adam 的学习率
            init_strategy: 新建 adapter 的初始化策略：
                           "warm"   = non-init 时从当前 active 复制权重（Day 1.5 默认）
                           "random" = 永远随机初始化（Day 2 confound-busting：
                                      解 fit_threshold vs warm-start 的双变量耦合）
                           init 时第一个 adapter 0 都走随机分支（self.adapters 为空时无 active 可复制）。
        """
        super().__init__()
        assert input_dim > 0
        assert hidden_dim > 0
        assert n_outputs > 0
        assert max_adapters >= 1
        assert fit_threshold > 0
        assert lr > 0
        assert init_strategy in ("warm", "random"), (
            f"init_strategy 必须 ∈ {{'warm', 'random'}}，收到: {init_strategy}"
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.max_adapters = max_adapters
        self.fit_threshold = fit_threshold
        self.lr = lr
        self.init_strategy = init_strategy

        # nn.ModuleDict 用 str 键
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        self._optimizers: dict = {}    # int → torch.optim.Adam
        self.usage: dict = {}          # int → 累计被 route 选中的次数
        self.route_history: list = []  # list[(t, active_id)]，由调用方填 t
        self._next_id: int = 0
        self._active_id: int = -1
        self.n_random_inits: int = 0     # 随机初始化的 adapter 计数（adapter 0 默认走这条）
        self.n_warmstart_inits: int = 0  # warm-start 自当前 active 复制的 adapter 计数

        # 默认创建 adapter 0 并 active（此时 self.adapters 为空 → 走随机初始化分支）
        self._create_new_adapter()

    # ------------------------------------------------------------------
    # 内部：构造与登记
    # ------------------------------------------------------------------

    def _make_mlp(self) -> nn.Sequential:
        """创建一个新的空白 MLP（与 GatedEnsemble.adapter 同结构）。"""
        mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_outputs),
        )
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return mlp

    def _create_new_adapter(self) -> int:
        """新建一个 adapter，返回其 id；自动设为 active。

        策略：
          - 第一次创建（init 时 self.adapters 为空）→ 随机初始化（cold-start adapter 0）
          - 之后所有 create（routing 触发时）→ 从当前 active adapter 复制权重 (warm-start)
            + 新建独立 Adam optimizer（state 自动重置）

        warm-start 解 cold-start 失败模式：v1 indicator 实验中 25/25 routing 全是
        新建空白 adapter，路由瞬间预测从训练好的 active 跳到随机初始化网络 →
        短期 acc 下降抵消 routing 增益。warm-start 让新 adapter 从已学到的状态出发
        继续在新 regime 上微调。
        """
        new_id = self._next_id
        self._next_id += 1
        mlp = self._make_mlp()  # 默认 Kaiming 随机初始化

        has_existing_active = (
            len(self.adapters) > 0
            and self._active_id >= 0
            and str(self._active_id) in self.adapters
        )
        do_warm = has_existing_active and self.init_strategy == "warm"
        if do_warm:
            # warm-start：从当前 active adapter 复制权重
            src = self.adapters[str(self._active_id)]
            with torch.no_grad():
                for p_dst, p_src in zip(mlp.parameters(), src.parameters()):
                    p_dst.copy_(p_src)
            self.n_warmstart_inits += 1
        else:
            # 随机初始化：(a) init 时 adapter 0；(b) init_strategy="random" 时所有 create
            self.n_random_inits += 1

        self.adapters[str(new_id)] = mlp
        # 新建独立 Adam → optimizer state（momentum、二阶矩）自动重置
        self._optimizers[new_id] = torch.optim.Adam(mlp.parameters(), lr=self.lr)
        self.usage[new_id] = 0
        self._active_id = new_id
        self.usage[new_id] += 1
        return new_id

    # ------------------------------------------------------------------
    # 核心接口：drop-in adapter
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """路由到当前 active adapter，返回 (B, n_outputs) 的 inter 残差预测。"""
        assert x.ndim == 2, f"x 应为 2D，收到 shape: {x.shape}"
        assert x.shape[-1] == self.input_dim, (
            f"x 特征维度应为 {self.input_dim}，收到 {x.shape[-1]}"
        )
        return self.adapters[str(self._active_id)](x)

    # ------------------------------------------------------------------
    # Routing：检测到漂移后由外部调用
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_existing(
        self,
        X_recent: torch.Tensor,
        errors_recent: torch.Tensor,
    ) -> dict:
        """
        对每个现有 adapter 计算 MSE(adapter(X_recent), errors_recent)。

        Args:
            X_recent:      (n, input_dim) tensor
            errors_recent: (n,) 或 (n, 1) tensor，目标残差信号

        Returns:
            dict[int, float]，adapter_id → MSE loss
        """
        assert X_recent.ndim == 2
        target = errors_recent.view(-1)
        assert X_recent.shape[0] == target.shape[0]

        losses: dict = {}
        for sid, mlp in self.adapters.items():
            pred = mlp(X_recent).view(-1)
            losses[int(sid)] = float(F.mse_loss(pred, target).item())
        return losses

    def route(
        self,
        X_recent: np.ndarray,
        errors_recent: np.ndarray,
        t: int = -1,
    ) -> tuple[int, str, dict]:
        """
        漂移触发后做路由决策。

        Args:
            X_recent:      (n, input_dim) 最近 n 步的原始特征
            errors_recent: (n,) 最近 n 步的 raw error（y_t - y_slow）
            t:             全局时间步坐标，用于 route_history 记录

        Returns:
            active_id:  路由后激活的 adapter id（int）
            action:     "switch" / "create" / "reuse"
                          - "switch":  从其他现有 adapter 切到一个更合适的现有 adapter
                          - "reuse":   评估后留在原 active（无切换，但仍记录路由事件）
                          - "create":  新建第 K+1 个 adapter
            losses:     评估各 adapter 的 dict[int, float]（含决策前各 adapter MSE）
        """
        assert isinstance(X_recent, np.ndarray)
        assert isinstance(errors_recent, np.ndarray)
        assert X_recent.ndim == 2 and X_recent.shape[1] == self.input_dim
        assert errors_recent.ndim == 1
        assert X_recent.shape[0] == errors_recent.shape[0]

        X_t = torch.tensor(X_recent, dtype=torch.float32)
        e_t = torch.tensor(errors_recent, dtype=torch.float32)

        losses = self.evaluate_existing(X_t, e_t)
        best_id = min(losses, key=losses.get)
        best_loss = losses[best_id]
        prev_active = self._active_id

        if best_loss <= self.fit_threshold or len(self.adapters) >= self.max_adapters:
            self._active_id = best_id
            self.usage[best_id] = self.usage.get(best_id, 0) + 1
            action = "switch" if best_id != prev_active else "reuse"
        else:
            self._create_new_adapter()  # 内部会更新 _active_id
            action = "create"

        self.route_history.append((int(t), int(self._active_id)))
        return int(self._active_id), action, losses

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def active_id(self) -> int:
        return self._active_id

    @property
    def active_adapter(self) -> nn.Module:
        return self.adapters[str(self._active_id)]

    def active_optimizer(self) -> torch.optim.Optimizer:
        """返回当前 active adapter 的 Adam optimizer（供 consolidation 使用）。"""
        return self._optimizers[self._active_id]

    def n_adapters(self) -> int:
        return len(self.adapters)

    def n_parameters(self) -> int:
        """所有 adapter 总可训练参数量（含非 active）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"AdapterLibrary("
            f"input_dim={self.input_dim}, "
            f"hidden={self.hidden_dim}, "
            f"n_adapters={self.n_adapters()}/{self.max_adapters}, "
            f"active={self._active_id}, "
            f"fit_threshold={self.fit_threshold}, "
            f"n_routes={len(self.route_history)})"
        )

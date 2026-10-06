"""
Phase 3B：快速校正器 → 中间层巩固（Fast→Inter Consolidation）

此模块属于 Phase 3B，负责将 FastCorrector 工作记忆中积累的
系统性误差模式蒸馏到 GatedEnsemble 的 adapter（inter 层）中，
完成"快→中"知识迁移，并在迁移完成后清空工作记忆。

巩固触发条件由调用方（FastCorrector.should_consolidate）决定，
本模块不重复实现判断逻辑，只负责执行蒸馏训练循环。

使用流程（prequential online 场景）：

    import torch
    from src.consolidation.fast_to_inter import FastToInterConsolidation
    from src.models.gated_ensemble import GatedEnsemble
    from src.models.fast_corrector import FastCorrector

    consolidator = FastToInterConsolidation(
        threshold=0.05,   # 保留，供诊断/日志使用
        window=50,
        epochs=10,
    )
    model = GatedEnsemble(input_dim=10)
    corrector = FastCorrector(buffer_size=200, method="knn")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 在主循环中，由 FastCorrector 判断是否到达触发条件
    if corrector.should_consolidate(window=50, bias_threshold=0.05):
        # X_recent: 最近 window 步的原始特征，numpy (window, n_features)
        consolidator.consolidate(
            gated_ensemble=model,
            fast_corrector=corrector,
            X_recent=X_recent,
            optimizer=optimizer,
        )
"""

import numpy as np
import torch
import torch.nn.functional as F


class FastToInterConsolidation:
    """
    快→中巩固器（Phase 3B）。

    功能：
        从 FastCorrector 的工作记忆中读取最近 window 步的预测误差，
        以此为监督信号训练 GatedEnsemble.adapter（inter 层），
        让 adapter 学会吸收 fast corrector 观察到的系统性漂移模式。
        训练完成后清空 FastCorrector 缓冲区，避免陈旧信息干扰下一轮。

    设计约束：
        - 不实现 should_consolidate —— 触发条件由 FastCorrector 提供
        - lr / epochs 控制内部训练循环，optimizer 由调用方传入（支持复用）
        - threshold / window 字段保留供诊断或日志记录使用
        - CPU 友好，无 GPU 依赖
    """

    def __init__(
        self,
        threshold: float = 0.05,
        window: int = 50,
        epochs: int = 10,
    ):
        """
        Args:
            threshold: 触发蒸馏的最小平均校正幅度（保留供日志/诊断用，不参与判断）
            window:    巩固观察窗口大小，即从 buffer 取最近多少步误差作为监督信号
            epochs:    每次巩固执行的梯度更新步数
        """
        assert threshold > 0, f"threshold 必须 > 0，收到: {threshold}"
        assert window >= 1, f"window 必须 >= 1，收到: {window}"
        assert epochs >= 1, f"epochs 必须 >= 1，收到: {epochs}"

        self.threshold = threshold
        self.window = window
        self.epochs = epochs

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def consolidate(
        self,
        gated_ensemble,
        fast_corrector,
        X_recent: np.ndarray,
        optimizer,
    ) -> float:
        """
        执行一次快→中巩固：让 adapter 拟合 fast corrector 的近期误差，
        完成后清空 fast corrector 的缓冲区。

        Args:
            gated_ensemble:  GatedEnsemble 实例，adapter 参数将被更新
            fast_corrector:  FastCorrector 实例，从其 buffer 读取误差，巩固后 reset
            X_recent:        (window, n_features) 最近 window 步的原始特征，numpy float
            optimizer:       PyTorch optimizer，已绑定 gated_ensemble.parameters()

        Returns:
            final_loss: 最后一个 epoch 的 MSE loss（float，供调用方记录）

        Raises:
            AssertionError: X_recent 样本数不等于 self.window，
                            或 buffer 中记录数少于 self.window
        """
        # ── 前置校验 ──────────────────────────────────────────────────
        assert isinstance(X_recent, np.ndarray), (
            f"X_recent 必须是 numpy ndarray，收到: {type(X_recent)}"
        )
        assert X_recent.ndim == 2, (
            f"X_recent 应为 2D 数组 (window, n_features)，收到 shape: {X_recent.shape}"
        )
        assert X_recent.shape[0] == self.window, (
            f"X_recent 的样本数必须等于 window={self.window}，"
            f"收到: {X_recent.shape[0]}"
        )
        assert len(fast_corrector.buffer) >= self.window, (
            f"fast_corrector.buffer 至少需要 {self.window} 条记录，"
            f"当前只有 {len(fast_corrector.buffer)} 条"
        )

        # ── 步骤 1：从 buffer 读取监督信号（公共 API，避免访问内部字段）──
        recent_errors: np.ndarray = fast_corrector.buffer.recent_errors(self.window)
        # recent_errors shape: (window,) float
        target = torch.tensor(recent_errors, dtype=torch.float32)  # (window,)

        # ── 步骤 2：将原始特征转为 tensor ────────────────────────────
        X_tensor = torch.tensor(X_recent, dtype=torch.float32)     # (window, n_features)

        # ── 步骤 3：训练循环 ──────────────────────────────────────────
        # 通过 get_inter_prediction 取 adapter 输出（蒸馏出口）
        # adapter 输出 shape: (window, n_outputs)；squeeze(-1) 压成 (window,) 对齐 target
        final_loss = 0.0
        gated_ensemble.train()
        for _ in range(self.epochs):
            y_inter = gated_ensemble.get_inter_prediction(X_tensor)  # (window, n_outputs)
            loss = F.mse_loss(y_inter.squeeze(-1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # ── 步骤 4：清空 fast corrector 缓冲区 ───────────────────────
        # reset() 内部调用 buffer.clear()，不直接操作 buffer 内部字段
        fast_corrector.reset()

        return final_loss

    def __repr__(self) -> str:
        return (
            f"FastToInterConsolidation("
            f"threshold={self.threshold}, "
            f"window={self.window}, "
            f"epochs={self.epochs})"
        )

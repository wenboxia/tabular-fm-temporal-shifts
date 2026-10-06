"""
Phase 3A：软门控融合网络（Soft Gating + Residual Adapter）

此模块属于 Phase 3A 的核心组件，负责将三个时间尺度的预测
（slow / inter / fast）通过可学习的软门控进行残差累加融合。

Phase 3 v2 (residual-additive fusion) 改进点：
  - 门控网络动态输出 [α, β, γ]（softmax 归一化，三者之和恒为 1）
  - inter 层由轻量 MLP（adapter）从**原始输入特征**学习残差校正
  - 输入是**原始特征 X**，不是 TabPFN 嵌入（V2 相对 V1 的明确改动）
  - 融合采用残差累加：y_final_raw = y_slow + β·y_inter + γ·correction
  - α 不参与融合计算，但 gate 输出维度保留为 3（保持 softmax 性质，
    向后兼容可视化；如需回切到 v1 加权融合，只需重新引入 alpha 项）

使用流程（prequential online 场景）：

    import torch
    from src.models.gated_ensemble import GatedEnsemble

    model = GatedEnsemble(input_dim=10, hidden_dim=64, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for batch in loader:
        x          = torch.tensor(batch.X_query, dtype=torch.float32)  # (B, D)
        y_slow     = torch.tensor(slow_proba,    dtype=torch.float32)  # (B, 1) 正类概率
        correction = torch.tensor(fast_corr,     dtype=torch.float32)  # (B, 1) 快速残差校正量

        y_final_raw, weights = model(x, y_slow, correction)   # (B, 1), (B, 3)
        loss = criterion(y_final_raw, y_true)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # weights[:, 0] = alpha (gate 维度保留，不参与融合)
        # weights[:, 1] = beta  (inter 权重)
        # weights[:, 2] = gamma (fast correction 权重)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedEnsemble(nn.Module):
    """
    软门控融合模块（Phase 3 v2, residual-additive fusion）。

    功能：
        1. gate 网络：以原始输入特征为条件，输出三路 softmax 权重 [α, β, γ]
        2. adapter 网络：以原始输入特征为条件，输出 inter 层的残差校正（即 y_inter）
        3. 残差累加融合：y_final_raw = y_slow + β·y_inter + γ·correction

    设计约束（最初设计）：
        - 输入是原始特征 X，不是 TabPFN 嵌入
        - TabPFN 权重绝不微调，本模块只学习 gate 和 adapter 参数
        - α + β + γ = 1（softmax 保证），防止数值饱和
        - α 不参与融合，但 gate 输出维度保留为 3（向后兼容可视化 + softmax
          性质便于回切）
        - 轻量 MLP，CPU 友好，无 GPU 依赖
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_outputs: int = 1,
    ):
        """
        Args:
            input_dim:  输入特征维度（原始特征数，即 X.shape[-1]）
            hidden_dim: gate 和 adapter 两个 MLP 的隐藏层宽度
            n_outputs:  输出维度（二分类场景取 1，即正类概率）
        """
        assert input_dim > 0, f"input_dim 必须 > 0，收到: {input_dim}"
        assert hidden_dim > 0, f"hidden_dim 必须 > 0，收到: {hidden_dim}"
        assert n_outputs > 0, f"n_outputs 必须 > 0，收到: {n_outputs}"

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs

        # 门控网络：原始特征 -> 3 个未归一化 logit -> softmax -> [α, β, γ]
        # α: slow 权重，β: inter 权重，γ: fast 权重
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),   # 3 = slow / inter / fast
        )

        # 中间层适配器：原始特征 -> inter 残差校正（无界，可正可负）
        # 不加 sigmoid，由训练阶段的 loss + softmax 门控权重平衡量纲
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        权重初始化：
          - 线性层：Kaiming 均匀初始化（适合 ReLU 激活）
          - 偏置：全零
          - gate 最后一层偏置全零，初始时三路权重均等（各约 1/3）
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y_slow: torch.Tensor,
        correction: torch.Tensor,
    ):
        """
        前向计算：残差累加融合三路预测（Phase 3 v2）。

        Args:
            x:          (batch_size, input_dim) 原始输入特征（float32）
            y_slow:     (batch_size, n_outputs) 慢速预测（TabPFN 正类概率，float32）
            correction: (batch_size, n_outputs) 快速残差校正量（FastCorrector 输出，float32）

        Returns:
            y_final_raw: (batch_size, n_outputs) 未 clamp 的融合预测（调用方负责 clamp）
            weights:     (batch_size, 3)         门控权重 [α, β, γ]，每行和为 1
        """
        assert x.ndim == 2, (
            f"x 应为 2D 张量 (batch_size, input_dim)，收到 shape: {x.shape}"
        )
        assert x.shape[-1] == self.input_dim, (
            f"x 的特征维度应为 {self.input_dim}，收到: {x.shape[-1]}"
        )
        assert y_slow.shape == correction.shape, (
            f"y_slow 和 correction 的 shape 必须一致，"
            f"收到 y_slow={y_slow.shape}, correction={correction.shape}"
        )

        # 门控权重：softmax 保证 α + β + γ = 1
        gate_logits = self.gate(x)                          # (B, 3)
        weights = F.softmax(gate_logits, dim=-1)            # (B, 3)
        beta  = weights[:, 1:2]                             # (B, 1)
        gamma = weights[:, 2:3]                             # (B, 1)
        # alpha = weights[:, 0:1] 保留在 weights 中供可视化，不参与融合计算

        # inter 层预测（adapter 直接以原始特征为输入）
        y_inter = self.adapter(x)                           # (B, n_outputs)

        # 残差累加融合（v2）：slow 为基础，beta·inter + gamma·correction 为增量
        y_final_raw = y_slow + beta * y_inter + gamma * correction  # (B, n_outputs)

        return y_final_raw, weights

    # ------------------------------------------------------------------
    # 状态查询与工具方法
    # ------------------------------------------------------------------

    def get_gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅返回门控权重，不执行完整前向计算（用于可视化/调试）。

        Args:
            x: (batch_size, input_dim) 原始输入特征

        Returns:
            weights: (batch_size, 3)，列顺序为 [α_slow, β_inter, γ_fast]
        """
        assert x.ndim == 2, (
            f"x 应为 2D 张量，收到 shape: {x.shape}"
        )
        with torch.no_grad():
            gate_logits = self.gate(x)
            return F.softmax(gate_logits, dim=-1)

    def get_inter_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅返回 adapter 的 inter 预测（用于快→中巩固时的蒸馏目标）。

        Args:
            x: (batch_size, input_dim) 原始输入特征

        Returns:
            y_inter: (batch_size, n_outputs)，无界残差校正量
        """
        assert x.ndim == 2, (
            f"x 应为 2D 张量，收到 shape: {x.shape}"
        )
        return self.adapter(x)

    def n_parameters(self) -> int:
        """返回可训练参数总量（含 gate 和 adapter）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"GatedEnsemble("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"n_outputs={self.n_outputs}, "
            f"n_params={self.n_parameters()})"
        )

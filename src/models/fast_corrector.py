"""
Level 3: 快速校正器

基于 FIFO 工作记忆缓冲区，对 TabPFN 的预测进行即时残差补偿。

两种校正策略（均无可训练参数，CPU 友好）：
  - "knn":  K 近邻误差均值（对局部分布变化更敏感）
  - "ema":  指数移动平均误差（对全局漂移趋势更平滑）

使用流程（prequential）：
    corrector = FastCorrector(buffer_size=100, method="knn")

    for batch in loader:
        # 1. TabPFN 给出慢速预测
        y_slow_prob = slow_prior.predict_proba(...)
        y_slow = y_slow_prob[:, 1]   # 取正类概率

        # 2. 快速校正
        correction = corrector.correct(x_query)
        y_pred_prob = np.clip(y_slow + correction, 0, 1)

        # 3. 观测真实标签后更新缓冲区
        error = y_true - y_slow      # 真实值 - 慢速预测（概率域）
        corrector.update(x_query, error)
"""

import numpy as np

from src.memory.buffer import WorkingMemoryBuffer


class FastCorrector:
    """
    无参数快速残差校正器。

    将最近的预测误差存入工作记忆缓冲区，
    每步通过 KNN 或 EMA 估计当前误差并补偿。
    """

    def __init__(
        self,
        buffer_size: int = 100,
        method: str = "knn",
        k: int = 5,
        alpha: float = 0.1,
    ):
        """
        Args:
            buffer_size: 工作记忆容量
            method:      "knn" 或 "ema"
            k:           KNN 的近邻数量
            alpha:       EMA 的平滑系数
        """
        assert method in ("knn", "ema"), f"method 必须是 'knn' 或 'ema'，收到: {method}"
        self.buffer = WorkingMemoryBuffer(capacity=buffer_size)
        self.method = method
        self.k = k
        self.alpha = alpha

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def correct(self, x_query: np.ndarray) -> float:
        """
        根据工作记忆估计当前时步的校正量。

        Args:
            x_query: (n_features,) 当前查询特征（单个样本）

        Returns:
            校正量（概率域标量），应加到 y_slow 上
        """
        if self.buffer.is_empty():
            return 0.0

        if self.method == "knn":
            return self.buffer.knn_correction(
                np.asarray(x_query).ravel(), k=self.k
            )
        else:
            return self.buffer.ema_correction(alpha=self.alpha)

    def update(self, x: np.ndarray, error: float) -> None:
        """
        观测到真实标签后，将本步的特征和误差压入缓冲区。

        Args:
            x:     (n_features,) 当前特征
            error: 真实值 - 慢速预测（概率域），即需要补偿的量
        """
        self.buffer.push(np.asarray(x).ravel(), error)

    def reset(self) -> None:
        """清空缓冲区（体制切换或巩固后调用）。"""
        self.buffer.clear()

    # ------------------------------------------------------------------
    # 状态查询（供巩固机制使用）
    # ------------------------------------------------------------------

    def should_consolidate(
        self,
        window: int = 50,
        bias_threshold: float = 0.05,
    ) -> bool:
        """
        判断是否需要触发快→中巩固。

        条件：缓冲区中最近 window 步的误差均值绝对值 > bias_threshold。

        B: 删除 std < |mean| 条件 —— 该规则对二分类 buffer errors 结构性过严，
        4 组消融全部 0 触发即证。改为只看 |mean| 绝对值。

        Args:
            window:          观察窗口
            bias_threshold:  触发巩固的最小偏置幅度

        Returns:
            True 表示应触发巩固
        """
        if len(self.buffer) < window:
            return False

        mean_err = self.buffer.mean_recent_error(window)
        std_err = self.buffer.std_recent_error(window)

        return abs(mean_err) > bias_threshold

    def __repr__(self) -> str:
        return (
            f"FastCorrector(method={self.method}, "
            f"buffer_size={self.buffer.capacity}, "
            f"current_size={len(self.buffer)})"
        )

"""
Level 1: 慢速先验 —— 冻结的 TabPFN 包装器

永远不微调 TabPFN 权重。
每个时间步接受一个上下文窗口（X_ctx, y_ctx）和查询样本 X_query，
通过 in-context learning 给出预测概率。

使用注意事项：
  - TabPFN 对 context_size 有上限（默认 10000），实际建议 ≤ 3000（CPU 友好）。
  - TabPFN 每次 fit 都会在内存中存储上下文，每步调用开销主要来自前向传播。
  - 本模块不依赖 GPU，可在 CPU 上运行。
"""

import warnings
from typing import Tuple

import numpy as np


class SlowPrior:
    """
    冻结的 TabPFN 包装器。

    对外接口：
        predict(X_ctx, y_ctx, X_query) -> (proba, pred_label)
    """

    def __init__(self, device: str = "auto", n_estimators: int = 8,
                 random_state: "int | None" = None):
        """
        Args:
            device: 'cpu' 或 'cuda'
            n_estimators: TabPFN 内部集成数量，越大越慢越准
                          CPU 上建议 4~8
            random_state: 传给 TabPFN 的随机种子。None = 沿用 TabPFN 默认值
                          （round 1 及以前都是 None，即 TabPFN 内部固定为 0，seed 从未改变过它）
        """
        from src.utils.device import resolve_device
        self.device = resolve_device(device)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._is_fitted = False

    def _get_model(self):
        """懒加载 TabPFN，避免 import 时就下载权重。"""
        if self._model is None:
            try:
                from tabpfn import TabPFNClassifier
            except ImportError as e:
                raise ImportError(
                    "请先安装 tabpfn: pip install tabpfn"
                ) from e
            kwargs = dict(device=self.device, n_estimators=self.n_estimators)
            if self.random_state is not None:
                kwargs["random_state"] = int(self.random_state)
            self._model = TabPFNClassifier(**kwargs)
        return self._model

    def predict(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        给定上下文和查询样本，返回预测概率和预测标签。

        Args:
            X_ctx:   (context_size, n_features) 上下文特征
            y_ctx:   (context_size,) 上下文标签（0/1 二分类）
            X_query: (batch_size, n_features) 查询特征

        Returns:
            proba:       (batch_size, n_classes) 每类的预测概率
            pred_labels: (batch_size,) 预测的类别（argmax）
        """
        model = self._get_model()

        # TabPFN 要求上下文至少包含两个类别
        unique_classes = np.unique(y_ctx)
        if len(unique_classes) < 2:
            # 极端情况：上下文只有一类，直接返回多数类
            majority = unique_classes[0]
            n_query = len(X_query)
            proba = np.zeros((n_query, 2))
            proba[:, int(majority)] = 1.0
            return proba, np.full(n_query, majority, dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_ctx, y_ctx)
            proba = model.predict_proba(X_query)

        if not np.all(np.isfinite(proba)):
            # GPU 半精度溢出等情况会产生 NaN；argmax(NaN) 会静默给出 0 类，结果看似正常实则作废。
            raise FloatingPointError(
                f"TabPFN 输出了非有限的概率（device={self.device}）：本次运行结果不可信，已停止"
            )
        pred_labels = np.argmax(proba, axis=1)
        return proba, pred_labels

    def predict_proba(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """仅返回概率（convenience wrapper）。"""
        proba, _ = self.predict(X_ctx, y_ctx, X_query)
        return proba

    def predict_label(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """仅返回预测标签（convenience wrapper）。"""
        _, pred_labels = self.predict(X_ctx, y_ctx, X_query)
        return pred_labels

    # ------------------------------------------------------------------
    # Round 2：多分类（Insects 原生 6 类）
    # ------------------------------------------------------------------

    def predict_proba_global(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
        n_classes: int,
        chunk: int = 1024,
    ) -> np.ndarray:
        """返回 (n_query, n_classes) 的概率，列 = 全局类别 0..n_classes-1。

        与二分类的 `predict` 不同，这里正确处理两件事：
          1. context 里缺某些类时，TabPFN 的 `predict_proba` 只输出 `classes_` 那几列，
             直接 argmax 得到的是**列位置**而不是类别标签。这里按 `classes_` 映射回全局列。
          2. context 只有一类时 TabPFN 无法拟合，返回该类的 one-hot（K 列，而不是 2 列）。
        查询按 `chunk` 分块预测；TabPFN 的测试行之间互不 attend，分块不改变结果。
        """
        y_ctx = np.asarray(y_ctx).astype(int)
        n_q = len(X_query)
        out = np.zeros((n_q, n_classes), dtype=np.float64)
        if n_q == 0:
            return out
        if len(y_ctx) == 0:
            raise ValueError("context 为空")
        if y_ctx.min() < 0 or y_ctx.max() >= n_classes:
            raise ValueError(f"context 标签超出 0..{n_classes - 1}")
        present = np.unique(y_ctx)
        if len(present) < 2:
            out[:, int(present[0])] = 1.0
            return out

        model = self._get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_ctx, y_ctx)
            cols = np.asarray(model.classes_).astype(int)
            for s in range(0, n_q, chunk):
                p = model.predict_proba(X_query[s: s + chunk])
                out[s: s + chunk, cols] = p
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(
                f"TabPFN 输出了非有限的概率（device={self.device}）：本次运行结果不可信，已停止"
            )
        return out

"""
Phase 3C：三时间尺度编排器（MultiTimescaleModel）

此模块属于 Phase 3C，将 SlowPrior / FastCorrector / GatedEnsemble /
FastToInterConsolidation 四个子模块串联成单一 prequential 接口，
每步调用 step() 完成：慢层预测 → 快层校正 → 门控融合 → 在线训练 → buffer 更新 → 按需巩固。

Phase 3 v2: residual-additive fusion (option A)
融合数学：y_final_raw = y_slow + β·y_inter + γ·correction
  correction 全功率参与，仅由 γ 控制；α 不参与 fusion 但 gate 输出维度保留为 3。

偏离最初设计的两处决策（在 MultiTimescaleModel 类 docstring 中说明）：
  [决策 1] 每步以 MSE 训练 gate + adapter
  [决策 2] y_final clamp 到 [0, 1] 后再作为预测输出

F: gate / adapter 分离 optimizer + consolidation cooldown，
解决 B 暴露的 adapter thrashing 问题（每步 backward 反复改写 adapter，
consolidation 学到的中期偏置被冲掉；同时巩固在稳定期反复触发）。

使用流程（prequential online 场景）：

    import numpy as np
    from src.models.multi_timescale import MultiTimescaleModel

    model = MultiTimescaleModel(input_dim=10)

    for batch in loader:
        x_t   = batch.X_query[0]           # (n_features,)  1D
        y_t   = float(batch.y_query[0])    # 标量 0/1
        X_ctx = batch.X_ctx                 # (context_size, n_features)
        y_ctx = batch.y_ctx                 # (context_size,)

        y_pred, weights = model.step(X_ctx, y_ctx, x_t, y_t, t=batch.t)
        # y_pred:  float ∈ [0, 1]，正类概率
        # weights: np.ndarray shape (3,)，[α_slow, β_inter, γ_fast]
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.slow_prior import SlowPrior
from src.models.fast_corrector import FastCorrector
from src.models.gated_ensemble import GatedEnsemble
from src.consolidation.fast_to_inter import FastToInterConsolidation
from src.drift.error_detector import (
    ADWINErrorDetector,
    RiverADWINDetector,
    make_detector,
)
from src.regime.adapter_library import AdapterLibrary

# Phase 5.5：报警后的动作分支（判别性对照的自变量）与触发源
ACTIONS_ON_ALARM = ("route_adapter", "context_reset", "buffer_clear", "none")
TRIGGER_SOURCES = ("detector", "oracle")

# Phase 5.5：喂给检测器的信号。scripts/diag_contrast_signal.py 的诊断实测：
#   indicator     现状，官方漂移召回 1/2（漏 d3_33240）
#   pred1         模型输出的类先验，召回 2/2，**不需要第二路预测**，d3 位移最大 0.340
#   contrast_prob 对比信号，|p_stale − p_sliding|，召回 2/2，d2 位移 0.491（indicator 的 2×）
#   contrast_hard 同上但取硬预测是否不一致
# 四者在无漂移的 d0_control 段上都是零误报。
DETECTOR_INPUTS = ("indicator", "pred1", "contrast_prob", "contrast_hard")


class MultiTimescaleModel:
    """
    三时间尺度编排器（Phase 3C）。

    功能：
        将 SlowPrior（Level 1）、FastCorrector（Level 3）、
        GatedEnsemble（Phase 3A 门控融合）、FastToInterConsolidation（Phase 3B）
        组合为统一的 prequential 接口，每步 step() 按固定顺序执行六步流程。

    偏离最初设计的决策：
        [决策 1] 每步 MSE 训练：step() 在观测真实标签 y_t 后，
            对 y_final_raw（未 clamp）计算 MSE(y_final_raw, y_t)，
            执行 backward + optimizer.step()，在线更新 gate 和 adapter 参数。
            选用 MSE 而非 BCE：adapter 输出无界，y_final 偶尔超出 [0,1]，
            MSE 对值域宽容；BCE 在 log(0) 附近数值不稳定。
        [决策 2] y_final clamp 后作为预测输出：
            clamp 发生在 MSE 计算之后（loss 使用原始 y_final_raw），
            返回值使用 torch.clamp(y_final_raw, 0, 1).item()，
            确保外部调用方拿到合法概率。

    Phase 3 v2 融合数学（residual-additive fusion, option A）：
        y_final_raw = y_slow + β·y_inter + γ·correction
        correction 全功率参与，仅由 γ 控制；α 不参与 fusion。
    """

    def __init__(
        self,
        input_dim: int,
        buffer_size: int = 100,
        fast_method: str = "knn",
        knn_k: int = 5,
        ema_alpha: float = 0.15,
        consolidation_threshold: float = 0.05,
        consolidation_window: int = 50,
        consolidation_epochs: int = 10,
        consolidation_cooldown: int = 100,
        gate_hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "auto",
        n_estimators: int = 4,
        # ── Phase 4 A 扩展（默认 False = 完全等价 Phase 3 v2+B+F）──
        use_adapter_library: bool = False,
        max_adapters: int = 8,
        library_fit_threshold: float = 0.5,
        library_init_strategy: str = "warm",
        detector_delta: float = 0.002,
        detector_min_subwindow: int = 30,
        detector_cooldown: int = 80,
        detector_impl: str = "own",
        detector_clock: int = 1,
        detector_input: str = "indicator",
        # ── Phase 5.5：报警动作策略（默认值 = Phase 4 A 既有行为）──
        action_on_alarm: "str | None" = None,
        trigger_source: str = "detector",
        oracle_trigger_times: "list[int] | None" = None,
        reset_size: int = 50,
        min_context_after_reset: int = 20,
        clear_detector_on_alarm: bool = True,
        consolidate_on_post_alarm_data: bool = False,
    ):
        """
        Args:
            input_dim:               原始特征维度（必须与数据集一致）
            buffer_size:             FastCorrector 工作记忆容量
            fast_method:             "knn" 或 "ema"
            knn_k:                   KNN 近邻数
            ema_alpha:               EMA 平滑系数
            consolidation_threshold: 触发巩固的最小平均误差绝对值
            consolidation_window:    巩固观察窗口（同时是 FastToInterConsolidation.window）
            consolidation_epochs:    每次巩固的梯度更新步数
            consolidation_cooldown:  两次巩固之间的最小间隔步数（防 thrashing）
            gate_hidden_dim:         GatedEnsemble gate/adapter 隐藏层宽度
            lr:                      Adam 学习率（gate optimizer 与 adapter optimizer 共用）
            device:                  "auto"（有 NVIDIA 显卡用 cuda，否则 cpu；不支持 mps）
            n_estimators:            TabPFN 集成数量
            use_adapter_library:     Phase 4 A 开关。False（默认）= Phase 3 v2+B+F 行为；
                                     True = 启用 ADWIN 检测 + 替换 GatedEnsemble.adapter 为 AdapterLibrary
            max_adapters:            AdapterLibrary 容量上限（仅 use_adapter_library=True 时生效）
            library_fit_threshold:   AdapterLibrary route 时复用现有 adapter 的 MSE 上限
            detector_delta:          ADWIN 置信参数（越小越保守）
            detector_min_subwindow:  ADWIN 切点两侧最小子窗
            detector_cooldown:       ADWIN 漂移声明后冷却步数
            detector_impl:           "own"（默认，自写 Hoeffding 版 = Phase 4/5 既有行为）
                                     或 "river"（标准 ADWIN，经验方差界；Phase 5.5）
            detector_clock:          river ADWIN 每隔多少步检查一次（1 = 每步；仅 river 生效）
            detector_input:          喂给检测器的信号，见 DETECTOR_INPUTS。
                                     "indicator"（默认 = 既有行为）/ "pred1" /
                                     "contrast_prob" / "contrast_hard"。
                                     contrast_* 需先调 set_stale_proba() 注入 stale 路概率。
            action_on_alarm:         报警后执行什么动作（Phase 5.5 判别性对照的自变量）：
                                     "route_adapter"（默认 = Phase 4 A 行为：路由 + 巩固 active adapter）
                                     / "context_reset"（截断 TabPFN 的滑动 context，Day 0.5 oracle 那一招）
                                     / "buffer_clear"（只清空 FastCorrector buffer）
                                     / "none"（只记录报警，不动作；纯消融对照）
            trigger_source:          "detector"（默认）或 "oracle"（用已知漂移点即时触发）。
                                     oracle 模式下 detector 仍在**影子模式**运行并记录 detector_events，
                                     用于测量检测延迟；且**不会**被 clear()，否则延迟测不准。
            oracle_trigger_times:    trigger_source="oracle" 时的触发时刻（全局 t 坐标）。
                                     为空会直接报错，避免静默退化成"永不适应"。
            reset_size:              context_reset 动作截断后的起始 context 长度（之后按
                                     min(reset_size + (t - alarm_t), len(X_ctx)) 平滑长回去，
                                     与 scripts/run_baselines.py --oracle_context_reset 同一时刻表）
            min_context_after_reset: 截断后的最小 context 长度（防单类 context 触发
                                     SlowPrior 常量 fallback → indicator 尖峰 → 误报循环）
            clear_detector_on_alarm: 路由后是否 detector.clear()。True = Phase 4 A 既有行为。
                                     oracle 模式下强制为 False（影子检测器必须保持独立）。
            consolidate_on_post_alarm_data:
                                     False（默认 = 既有行为）：报警即用 buffer 里最近
                                     consolidation_window 个样本巩固 —— 但这些样本可能**全在切点之前**
                                     （oracle 即时触发时尤其如此），等于用旧概念数据训练新 adapter。
                                     True：把巩固推迟到 alarm_t + consolidation_window，
                                     确保训练样本全部来自报警之后。
        """
        assert input_dim > 0, f"input_dim 必须 > 0，收到: {input_dim}"
        from src.utils.device import resolve_device
        device = resolve_device(device)
        self.device = device

        # ── Phase 5.5：报警动作策略校验 ────────────────────────────────
        # action_on_alarm=None（默认）自动解析：开了 library → Phase 4 A 的 route_adapter；
        # 没开 library（Phase 3 路径）→ "none"，动作分派整段被跳过，行为与改动前完全一致。
        if action_on_alarm is None:
            action_on_alarm = "route_adapter" if use_adapter_library else "none"
        assert action_on_alarm in ACTIONS_ON_ALARM, (
            f"action_on_alarm 必须 ∈ {ACTIONS_ON_ALARM}，收到: {action_on_alarm!r}"
        )
        assert trigger_source in TRIGGER_SOURCES, (
            f"trigger_source 必须 ∈ {TRIGGER_SOURCES}，收到: {trigger_source!r}"
        )
        if action_on_alarm == "route_adapter" and not use_adapter_library:
            raise ValueError(
                "action_on_alarm='route_adapter' 需要 use_adapter_library=True"
                "（Phase 3 路径没有 AdapterLibrary 可路由）"
            )
        if trigger_source == "oracle":
            if not oracle_trigger_times:
                # 静默的空 oracle 会让整个 run 退化成"永不适应"，却写出看似正常的 npz
                raise ValueError(
                    "trigger_source='oracle' 但 oracle_trigger_times 为空。"
                    "Electricity 没有 documented 漂移点；请显式传入触发时刻，"
                    "或改用 trigger_source='detector'。"
                )
            # 影子检测器必须独立于 oracle 动作，否则测不准检测延迟
            clear_detector_on_alarm = False
        assert detector_input in DETECTOR_INPUTS, (
            f"detector_input 必须 ∈ {DETECTOR_INPUTS}，收到: {detector_input!r}"
        )
        self.detector_input = detector_input
        self.detector_signal_history: list = []
        self._stale_proba = None          # contrast_* 用；由 set_stale_proba 注入
        self._stale_offset = 0
        assert reset_size > 0, f"reset_size 必须 > 0，收到: {reset_size}"
        assert min_context_after_reset > 0, (
            f"min_context_after_reset 必须 > 0，收到: {min_context_after_reset}"
        )

        self.action_on_alarm = action_on_alarm
        self.trigger_source = trigger_source
        self.oracle_trigger_times = sorted(int(x) for x in (oracle_trigger_times or []))
        self._oracle_set = set(self.oracle_trigger_times)
        self.reset_size = reset_size
        self.min_context_after_reset = min_context_after_reset
        self.clear_detector_on_alarm = clear_detector_on_alarm
        self.consolidate_on_post_alarm_data = consolidate_on_post_alarm_data
        self.alarm_events: list = []          # 实际驱动动作的报警时刻
        self.action_events: list = []         # list[(t, action)]，动作真正执行的时刻
        self._last_alarm_t: int | None = None
        self._pending_consolidation_t: int | None = None
        self.n_context_truncations: int = 0

        self.input_dim = input_dim
        self.consolidation_window = consolidation_window
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_cooldown = consolidation_cooldown
        self.use_adapter_library = use_adapter_library
        self._step_count: int = 0
        self._last_consolidation_t: float = -float('inf')
        self.consolidation_events: list = []
        self.detector_events: list = []           # 仅 use_adapter_library=True 时记录
        self.route_events: list = []              # list[(t, action, active_id)]
        self.abs_error_history: list = []         # |error| 时序，diagnostic 用
        self.indicator_history: list = []         # 0/1 错误指示器时序（option B detector 输入）

        # ── 子模块初始化 ──────────────────────────────────────────────
        self.slow_prior = SlowPrior(device=device, n_estimators=n_estimators)

        self.fast_corrector = FastCorrector(
            buffer_size=buffer_size,
            method=fast_method,
            k=knn_k,
            alpha=ema_alpha,
        )

        self.gated_ensemble = GatedEnsemble(
            input_dim=input_dim,
            hidden_dim=gate_hidden_dim,
            n_outputs=1,
        )

        self.consolidator = FastToInterConsolidation(
            threshold=consolidation_threshold,
            window=consolidation_window,
            epochs=consolidation_epochs,
        )

        # ── 优化器：仅优化 GatedEnsemble（TabPFN 权重绝不微调）─────────
        self.gate_optimizer = torch.optim.Adam(
            self.gated_ensemble.gate.parameters(), lr=lr
        )

        # ── Phase 4 A：可选启用 AdapterLibrary + ADWINErrorDetector ──
        # 默认 use_adapter_library=False 时走 Phase 3 v2+B+F 路径：
        #   self.gated_ensemble.adapter 为单一 nn.Sequential，
        #   self.adapter_optimizer 为该单一 adapter 的 Adam。
        # 开启后：
        #   self.gated_ensemble.adapter 被替换为 AdapterLibrary 实例（drop-in），
        #   self.adapter_optimizer 设为 None（consolidate 时改用 library.active_optimizer()），
        #   self.detector 为 ADWIN 实例，每步喂 raw error。
        self.detector: ADWINErrorDetector | RiverADWINDetector | None = None
        self.detector_impl = detector_impl
        self.adapter_library: AdapterLibrary | None = None
        if use_adapter_library:
            self.adapter_library = AdapterLibrary(
                input_dim=input_dim,
                hidden_dim=gate_hidden_dim,
                n_outputs=1,
                max_adapters=max_adapters,
                fit_threshold=library_fit_threshold,
                lr=lr,
                init_strategy=library_init_strategy,
            )
            self.gated_ensemble.adapter = self.adapter_library  # drop-in
            # Phase 5.5：detector_impl="river" 换标准 ADWIN（经验方差界）。
            # own 版的 max_window=buffer_size*4=400 + value_range=1.0 使 200/200 切分
            # 也要求 |Δmean| ≥ 0.209，真实 Insects indicator 位移 ~0.10 结构性触发不了。
            self.detector = make_detector(
                detector_impl,
                delta=detector_delta,
                min_subwindow=detector_min_subwindow,
                max_window=max(2 * detector_min_subwindow, buffer_size * 4),
                value_range=1.0,        # |error| / indicator ∈ [0, 1]
                cooldown=detector_cooldown,
                clock=detector_clock,
            )
            self.adapter_optimizer = None
        else:
            self.adapter_optimizer = torch.optim.Adam(
                self.gated_ensemble.adapter.parameters(), lr=lr
            )

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def step(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        x_t: np.ndarray,
        y_t: float,
        t: int = -1,
    ):
        """
        执行单步 prequential 推理与更新。

        Args:
            X_ctx: (context_size, input_dim) 上下文特征，喂给 TabPFN
            y_ctx: (context_size,) 上下文标签（0/1）
            x_t:   (input_dim,) 当前时步查询特征（1D）
            y_t:   当前时步真实标签（0 或 1，标量）
            t:     全局时间步坐标，由调用方传入（对应数据集原始下标，从 context_size 起步）

        Returns:
            y_pred:  float ∈ [0, 1]，clamp 后的正类概率（可直接用于 ≥0.5 判断）
            weights: np.ndarray shape (3,)，门控权重 [α_slow, β_inter, γ_fast]

        v2 融合数学：y_final_raw = y_slow + β·y_inter + γ·correction
        correction 全功率参与，仅由 γ 控制；clamp 在返回前统一做。

        consolidation_events 存的是全局时间步坐标，由调用方通过 t 参数提供；
        consolidation 触发由 fast_corrector.should_consolidate 单点判断；
        consolidate() 内部有 assert 兜底形状，外层不加额外保护（YAGNI）。
        """
        assert isinstance(t, int), f"t 必须为 int，收到: {type(t)}"
        assert X_ctx.ndim == 2, (
            f"X_ctx 应为 2D 数组 (context_size, input_dim)，收到 shape: {X_ctx.shape}"
        )
        assert X_ctx.shape[1] == self.input_dim, (
            f"X_ctx 特征维度应为 {self.input_dim}，收到: {X_ctx.shape[1]}"
        )
        x_t = np.asarray(x_t, dtype=np.float32).ravel()
        assert x_t.shape[0] == self.input_dim, (
            f"x_t 长度应为 {self.input_dim}，收到: {x_t.shape[0]}"
        )
        y_t = float(y_t)

        # ── Step 0：oracle 报警 + 预测前动作（Phase 5.5）──────────────
        # oracle 报警必须在 predict 之前登记并生效，才能与
        # scripts/run_baselines.py --oracle_context_reset 用同一时刻表
        # （那里在 t == drift_point 当步就已截断）。detector 报警只能在观测标签之后
        # 产生，所以它的动作从下一步开始生效 —— 这是检测本身的固有延迟，不是 bug。
        if self.trigger_source == "oracle" and t in self._oracle_set:
            self._register_alarm(t)
        if self.action_on_alarm == "context_reset" and self._last_alarm_t is not None:
            X_ctx, y_ctx = self._apply_context_reset(X_ctx, y_ctx, t)

        # ── Step 1：慢层预测（TabPFN in-context learning）────────────
        X_query_2d = x_t[np.newaxis, :]                   # (1, input_dim)
        proba = self.slow_prior.predict_proba(X_ctx, y_ctx, X_query_2d)
        y_slow: float = float(proba[0, 1])                # 正类概率，标量

        # ── Step 2：快层校正 ─────────────────────────────────────────
        correction: float = self.fast_corrector.correct(x_t)

        # ── Step 3：门控融合（v2 residual-additive）─────────────────
        # correction 直接作为残差传入，不预先 clip；clamp 在返回前统一做
        x_tensor          = torch.tensor(x_t[np.newaxis, :], dtype=torch.float32)  # (1, D)
        y_slow_tensor     = torch.tensor([[y_slow]],          dtype=torch.float32)  # (1, 1)
        correction_tensor = torch.tensor([[correction]],      dtype=torch.float32)  # (1, 1)

        self.gated_ensemble.train()
        y_final_raw, weights_tensor = self.gated_ensemble(
            x_tensor, y_slow_tensor, correction_tensor
        )                                                  # (1,1), (1,3)

        # ── Step 4：每步 MSE 训练（偏离最初设计：决策 1）────────────────
        # Phase 3 F：per-step backward 仅 step gate；adapter 不动（其梯度需 zero 以防累积）。
        # Phase 4 A：use_adapter_library=True 时 adapter_optimizer=None，
        # adapter 的 zero_grad 改用 library.active_optimizer()（active adapter 的 Adam）。
        y_t_tensor = torch.tensor([[y_t]], dtype=torch.float32)            # (1, 1)
        loss = nn.functional.mse_loss(y_final_raw, y_t_tensor)
        self.gate_optimizer.zero_grad()
        if self.use_adapter_library:
            self.adapter_library.active_optimizer().zero_grad()
        else:
            self.adapter_optimizer.zero_grad()
        loss.backward()
        self.gate_optimizer.step()           # 仅 step gate；adapter 不动

        # ── Step 5：观测后更新 buffer ────────────────────────────────
        error: float = y_t - y_slow
        self.fast_corrector.update(x_t, error)

        # ── Step 6：按需巩固 ─────────────────────────────────────────
        # 触发逻辑：
        #   - use_adapter_library=False（Phase 3 v2+B+F）：fast_corrector 的
        #     bias-threshold + cooldown 触发 → consolidate 单一 adapter
        #   - use_adapter_library=True （Phase 4 A）：ADWIN detector 在 raw error 流上
        #     报警 → route + consolidate active adapter
        #     不再用 bias-threshold，避免与 detector 抢事件并清空 buffer。
        #     若 ADWIN 在某数据集上从不触发（如 rotating_boundary 渐进漂移），
        #     adapter 仅靠 per-step gate 训练 + frozen 初始化参与融合，符合 YAGNI。
        #
        # routing 后 detector.clear() 让 detector 从新 regime 重新积累。
        # buffer 在 consolidate() 内部统一被 reset。
        # 诊断信号始终落盘（与是否开 library、用哪种触发源无关），
        # 这样每个动作分支的 run 都能拿到同口径的 indicator 流做事后重放。
        self.abs_error_history.append(abs(error))
        y_pred_hard = 1 if torch.clamp(y_final_raw.detach(), 0.0, 1.0).item() >= 0.5 else 0
        indicator = int(y_pred_hard != int(y_t))
        self.indicator_history.append(indicator)

        # 检测器输入：indicator 之外的三种都不依赖真实标签，
        # contrast_* 需要调用方预先传入 stale 路概率（见 set_stale_proba）。
        if self.detector_input == "indicator":
            detector_signal = float(indicator)
        elif self.detector_input == "pred1":
            detector_signal = float(y_pred_hard)
        else:
            p_stale = self._stale_proba_at(t)
            if self.detector_input == "contrast_prob":
                detector_signal = abs(p_stale - y_slow)
            else:  # contrast_hard
                detector_signal = float((p_stale >= 0.5) != (y_slow >= 0.5))
        self.detector_signal_history.append(detector_signal)

        if self.detector is not None:
            # Option B: detector 输入 = 0/1 错误指示器。
            # raw error 流（mean≈0）和 |error| 流（mean≈0.30 across regimes）的 mean shift
            # 都被 TabPFN sliding-context 自适应消化掉，ADWIN 结构性看不到信号。
            # ⚠️ Phase 5.5：真实数据上 indicator 位移只有 ~0.10，自写 ADWIN 的值域 Hoeffding 界
            # 要求 ≥0.209 → 结构性触发不了。用 --detector_impl river 换经验方差界可触发。
            #
            # detector 始终运行；oracle 模式下它是**影子模式**（只记录 detector_events，
            # 不驱动动作、也不被 clear），这样同一个 run 里既有 oracle 动作效果，
            # 又有真实检测延迟可测。
            detector_drift = self.detector.update(detector_signal)
            if detector_drift:
                self.detector_events.append(t)
            if self.trigger_source == "detector" and detector_drift:
                self._register_alarm(t)
        else:
            detector_drift = False

        # 本步是否有报警驱动动作（oracle 报警在 Step 0 已登记）
        alarm_now = self._last_alarm_t == t

        if self.use_adapter_library:
            should_trigger = alarm_now
        else:
            # Phase 3 v2+B+F 路径：bias-threshold + cooldown（行为与 Phase 5.5 改动前一致）。
            # 若显式指定了 oracle / 非默认动作，alarm_now 也参与触发。
            should_trigger = alarm_now or (
                t - self._last_consolidation_t >= self.consolidation_cooldown
                and self.fast_corrector.should_consolidate(
                    window=self.consolidation_window,
                    bias_threshold=self.consolidation_threshold,
                )
            )

        # ── Step 6b：报警后的动作分派（Phase 5.5）─────────────────────
        # action_on_alarm="route_adapter" + trigger_source="detector" 是既有 Phase 4 A 路径；
        # use_adapter_library=False 时 should_trigger 来自 bias-threshold，走同一段巩固逻辑。
        if alarm_now:
            self.action_events.append((t, self.action_on_alarm))
            if self.action_on_alarm == "buffer_clear":
                self.fast_corrector.reset()
            elif self.action_on_alarm == "context_reset":
                # 动作本体在 Step 0 生效（下一步预测前截断 context），这里无事可做
                pass
            elif self.action_on_alarm == "none":
                pass

        # consolidate_on_post_alarm_data=True 时把巩固推迟到 alarm_t + consolidation_window，
        # 保证训练样本全部来自报警之后（oracle 即时触发时 buffer 里全是切点**之前**的样本，
        # 直接巩固等于用旧概念数据训练新 adapter）。
        do_route = (
            should_trigger
            and self.action_on_alarm == "route_adapter"
            and self.adapter_library is not None
        )
        if do_route and self.consolidate_on_post_alarm_data:
            self._pending_consolidation_t = t + self.consolidation_window
            do_route = False
        elif (
            self._pending_consolidation_t is not None
            and t >= self._pending_consolidation_t
        ):
            self._pending_consolidation_t = None
            do_route = True

        legacy_bias_trigger = should_trigger and not self.use_adapter_library

        if do_route or legacy_bias_trigger:
            buf_len = len(self.fast_corrector.buffer)
            if buf_len >= self.consolidation_window:
                X_recent = self.fast_corrector.buffer.recent_features(
                    self.consolidation_window
                )
                # 报警触发先做 routing：评估现有 / 新建 → 切 active
                if do_route and self.adapter_library is not None:
                    errs_recent = self.fast_corrector.buffer.recent_errors(
                        self.consolidation_window
                    )
                    active_id, action, losses = self.adapter_library.route(
                        X_recent, errs_recent, t=t,
                    )
                    self.route_events.append((t, action, active_id))
                    if self.clear_detector_on_alarm and self.detector is not None:
                        self.detector.clear()

                # consolidate：use_adapter_library=True 时用 active adapter 的 optimizer
                opt = (
                    self.adapter_library.active_optimizer()
                    if self.use_adapter_library
                    else self.adapter_optimizer
                )
                self.consolidator.consolidate(
                    gated_ensemble=self.gated_ensemble,
                    fast_corrector=self.fast_corrector,
                    X_recent=X_recent,
                    optimizer=opt,
                )
                self._last_consolidation_t = t
                self.consolidation_events.append(t)

        self._step_count += 1  # 保留用于 __repr__ 调试展示，不再用于事件记录

        # ── 返回：clamp 后概率（偏离最初设计：决策 2）+ 门控权重 ─────
        # clamp 发生在 MSE loss 计算之后，loss 使用的是 y_final_raw（无 clamp）
        y_pred: float = torch.clamp(y_final_raw.detach(), 0.0, 1.0).item()
        weights_np: np.ndarray = (
            weights_tensor.detach().squeeze(0).numpy()     # (3,)
        )
        return y_pred, weights_np

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def reset_fast(self) -> None:
        """手动清空快速校正器缓冲区（如在已知漂移点处调用）。"""
        self.fast_corrector.reset()

    # ------------------------------------------------------------------
    # Phase 5.5：报警与动作
    # ------------------------------------------------------------------

    def set_stale_proba(self, proba: np.ndarray, offset: int = 0) -> None:
        """注入 stale 路（不适应）的正类概率，供 contrast_* 检测输入使用。

        stale 路的 context 固定为该段最早若干样本，**在 t=0 就已全部可得**，
        所以整段可以一次批量算完（实测 2–5 s，对比逐步滑窗的 304–801 s），
        再按 t 查表。这不是偷看未来：用到的标签全在 offset 之前。

        Args:
            proba:  (n,) 正类概率，proba[i] 对应全局时刻 offset + i
            offset: proba[0] 对应的全局 t
        """
        self._stale_proba = np.asarray(proba, dtype=np.float64)
        self._stale_offset = int(offset)

    def _stale_proba_at(self, t: int) -> float:
        if self._stale_proba is None:
            raise RuntimeError(
                f"detector_input={self.detector_input!r} 需要 stale 路概率，"
                "请先调用 set_stale_proba()"
            )
        i = t - self._stale_offset
        if not (0 <= i < len(self._stale_proba)):
            raise IndexError(
                f"t={t} 超出 stale_proba 覆盖范围 "
                f"[{self._stale_offset}, {self._stale_offset + len(self._stale_proba)})"
            )
        return float(self._stale_proba[i])

    def _register_alarm(self, t: int) -> None:
        """登记一次**驱动动作**的报警（区别于 detector_events：后者含影子模式的观测）。"""
        self._last_alarm_t = t
        self.alarm_events.append(t)

    def _apply_context_reset(
        self, X_ctx: np.ndarray, y_ctx: np.ndarray, t: int,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """截断 TabPFN 的 context 到报警之后的部分，长度按时间平滑长回全窗。

        时刻表与 scripts/run_baselines.py --oracle_context_reset 完全一致：
            k = min(reset_size + (t - alarm_t), len(X_ctx))

        两道守卫：
          1. k ≥ min_context_after_reset —— 太短的 context 会让 TabPFN 极不稳定；
          2. 截断后至少有 2 个类别 —— 否则 SlowPrior 走常量 fallback（proba 全 0/1），
             error 立刻变成 ±1、indicator 尖峰，反过来制造误报循环。
             不满足就成倍放宽 k，直到有 2 类或用尽整窗。
        """
        assert self._last_alarm_t is not None
        n = len(X_ctx)
        k = min(self.reset_size + (t - self._last_alarm_t), n)
        k = max(k, min(self.min_context_after_reset, n))
        if k >= n:
            return X_ctx, y_ctx

        X_r, y_r = X_ctx[-k:], y_ctx[-k:]
        while len(np.unique(y_r)) < 2 and k < n:
            k = min(k * 2, n)
            X_r, y_r = X_ctx[-k:], y_ctx[-k:]
        if k >= n:
            return X_ctx, y_ctx

        self.n_context_truncations += 1
        return X_r, y_r

    def __repr__(self) -> str:
        return (
            f"MultiTimescaleModel("
            f"input_dim={self.input_dim}, "
            f"step_count={self._step_count}, "
            f"consolidation_events={len(self.consolidation_events)}, "
            f"fast={self.fast_corrector.method})"
        )

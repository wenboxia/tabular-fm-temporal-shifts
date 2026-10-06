"""Round 2：决定每一步给 TabPFN 看哪些已揭示的行（"上下文记忆"）。

所有记忆只存**流内下标**，按时间升序；标签揭示后才通过 `update` 进入记忆，
所以任何上下文行都早于被预测的行（由 runner 再断言一次）。

统一接口：
    update(idx)                  新揭示的流内下标（升序）
    close_regime(rows)           一个阶段结束（只有档案类方法用到）
    contexts(X_query)            -> [(查询位置数组, 上下文下标数组, 路由编号)]
    stored_rows() / max_context() 预算统计
    digest()                     记忆状态的哈希（回测前后必须不变）
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

from src.data.temporal_loader import evict_oldest_of_majority
from src.memory.router import WINDOW, route_queries

NO_ROUTE = -2


def _hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


class SlidingMemory:
    """最近 W 行（W = 200 就是单独的 TabPFN 基线）。"""

    def __init__(self, window: int):
        self.window = int(window)
        self._q: deque = deque(maxlen=self.window)

    def update(self, idx) -> None:
        self._q.extend(int(i) for i in idx)

    def close_regime(self, rows) -> None:
        pass

    def window_idx(self) -> np.ndarray:
        return np.fromiter(self._q, dtype=np.int64, count=len(self._q))

    def contexts(self, X_query):
        return [(np.arange(len(X_query)), self.window_idx(), NO_ROUTE)]

    def stored_rows(self) -> int:
        return len(self._q)

    def max_context(self) -> int:
        return self.window

    def digest(self) -> str:
        return _hash(self.window_idx())


class ClassBalancedFIFO:
    """每类保留最近的 budget // K 行。"""

    def __init__(self, budget: int, y: np.ndarray, n_classes: int):
        self.per_class = int(budget) // int(n_classes)
        self.y = y
        self._q = [deque(maxlen=self.per_class) for _ in range(n_classes)]

    def update(self, idx) -> None:
        for i in idx:
            self._q[int(self.y[i])].append(int(i))

    def close_regime(self, rows) -> None:
        pass

    def _all(self) -> np.ndarray:
        return np.sort(np.fromiter((i for q in self._q for i in q), dtype=np.int64))

    def contexts(self, X_query):
        return [(np.arange(len(X_query)), self._all(), NO_ROUTE)]

    def stored_rows(self) -> int:
        return sum(len(q) for q in self._q)

    def max_context(self) -> int:
        return self.per_class * len(self._q)

    def digest(self) -> str:
        return _hash(self._all())


class DualMemory:
    """KDD 2026（Lourenço et al.）的双记忆：短期 FIFO + 类均衡长期池，无年龄上限。

    短期满了，最老的一条移入长期；长期满了，淘汰"数量最多那一类里最老的一条"
    （与 `DualMemoryLoader` 共用 `evict_oldest_of_majority`）。
    """

    def __init__(self, total: int, short_ratio: float, y: np.ndarray):
        self.short_cap = max(1, int(round(total * short_ratio)))
        self.long_cap = int(total) - self.short_cap
        assert self.long_cap >= 1
        self.y = y
        self._short: list = []
        self._long: list = []

    def update(self, idx) -> None:
        for i in idx:
            self._short.append(int(i))
            if len(self._short) > self.short_cap:
                self._long.append(self._short.pop(0))
                if len(self._long) > self.long_cap:
                    evict_oldest_of_majority(self._long, self.y)

    def close_regime(self, rows) -> None:
        pass

    def _all(self) -> np.ndarray:
        return np.asarray(self._long + self._short, dtype=np.int64)

    def contexts(self, X_query):
        return [(np.arange(len(X_query)), self._all(), NO_ROUTE)]

    def stored_rows(self) -> int:
        return len(self._short) + len(self._long)

    def max_context(self) -> int:
        return self.short_cap + self.long_cap

    def digest(self) -> str:
        return _hash(self._long, self._short)


class RegimeBalancedDual:
    """阶段均衡的双记忆（Round 2 修订后的"我们的方法"，见 phase56_prereg.md 修订 1）。

    结构与 KDD 2026 双记忆完全相同：短期 FIFO（total × short_ratio 行）+ 长期池，
    短期满了最老的一条移入长期池。唯一的区别是长期池满了时淘汰谁：
      - 双记忆：淘汰"数量最多的那一类"里最老的一条 → 长期池逐渐被最近几个阶段占满；
      - 本方法：先找长期池里行数最多的**阶段**（并列时取较新的阶段），再在该阶段里找
        行数最多的**类别**（并列时取编号小的），淘汰其中最老的一条。
    结果是每个过去的阶段在长期池里名额大致相等，阶段内类别也大致均衡。

    阶段由 `close_regime` 划分（官方变点或 ADWIN 报警），每行在揭示时记下当时的阶段号。
    没有任何 `close_regime` 调用时，行为与 `DualMemory` 完全相同。
    """

    def __init__(self, total: int, short_ratio: float, y: np.ndarray):
        self.short_cap = max(1, int(round(total * short_ratio)))
        self.long_cap = int(total) - self.short_cap
        assert self.long_cap >= 1
        self.y = y
        self._short: list = []
        self._long: list = []
        self._seg: dict = {}          # 行下标 -> 阶段号
        self._cur = 0

    def update(self, idx) -> None:
        for i in idx:
            i = int(i)
            self._seg[i] = self._cur
            self._short.append(i)
            if len(self._short) > self.short_cap:
                self._long.append(self._short.pop(0))
                if len(self._long) > self.long_cap:
                    self._evict()

    def _evict(self) -> None:
        seg_count: dict = {}
        for i in self._long:
            g = self._seg[i]
            seg_count[g] = seg_count.get(g, 0) + 1
        g_star = max(seg_count, key=lambda g: (seg_count[g], g))
        cls_count: dict = {}
        for i in self._long:
            if self._seg[i] == g_star:
                c = int(self.y[i])
                cls_count[c] = cls_count.get(c, 0) + 1
        c_star = max(cls_count, key=lambda c: (cls_count[c], -c))
        for pos, i in enumerate(self._long):          # 按时间升序，第一个命中的就是最老的
            if self._seg[i] == g_star and int(self.y[i]) == c_star:
                self._long.pop(pos)
                del self._seg[i]
                return

    def close_regime(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return
        cut = int(rows.max()) + 1
        self._cur += 1
        for i in self._short + self._long:            # 关闭点之后、已揭示的行属于新阶段
            if i >= cut:
                self._seg[i] = self._cur

    def _all(self) -> np.ndarray:
        return np.asarray(self._long + self._short, dtype=np.int64)

    def contexts(self, X_query):
        return [(np.arange(len(X_query)), self._all(), NO_ROUTE)]

    def stored_rows(self) -> int:
        return len(self._short) + len(self._long)

    def max_context(self) -> int:
        return self.short_cap + self.long_cap

    def segment_counts(self) -> dict:
        out: dict = {}
        for i in self._long:
            out[self._seg[i]] = out.get(self._seg[i], 0) + 1
        return out

    def digest(self) -> str:
        return _hash(self._long, self._short, [self._seg[i] for i in self._long + self._short], [self._cur])


class RegimeArchive:
    """滑窗 + 阶段档案。

    每个阶段结束时，从该阶段**全部**已揭示的行里按类别均衡随机抽 `per_class` 行存档。
    mode = "union"：所有档案都放进上下文；
    mode = "routed"：只有查询更像某个档案时才加入那一个档案（规则见 router.py）。
    档案数超过 `max_archives` 时淘汰最老的一个。
    """

    def __init__(self, window: int, per_class: int, mode: str, X: np.ndarray, y: np.ndarray,
                 n_classes: int, rng: np.random.Generator, k: int = 10, max_archives: int = 8):
        if mode not in ("union", "routed"):
            raise ValueError(mode)
        self.win = SlidingMemory(window)
        self.per_class = int(per_class)
        self.mode = mode
        self.X = X
        self.y = y
        self.n_classes = int(n_classes)
        self.rng = rng
        self.k = int(k)
        self.max_archives = int(max_archives)
        self.archives: "list[np.ndarray]" = []

    def update(self, idx) -> None:
        self.win.update(idx)

    def close_regime(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return
        keep = []
        for c in range(self.n_classes):
            rc = rows[self.y[rows] == c]
            if len(rc) == 0:
                continue
            take = min(self.per_class, len(rc))
            keep.append(self.rng.choice(rc, size=take, replace=False))
        self.archives.append(np.sort(np.concatenate(keep)))
        if len(self.archives) > self.max_archives:
            self.archives.pop(0)

    def contexts(self, X_query):
        w = self.win.window_idx()
        if self.mode == "union" or not self.archives:
            ctx = np.concatenate(self.archives + [w]) if self.archives else w
            return [(np.arange(len(X_query)), ctx, NO_ROUTE if self.mode == "union" else WINDOW)]
        routes = route_queries(X_query, self.X[w], [self.X[a] for a in self.archives], k=self.k)
        groups = []
        for r in np.unique(routes):
            pos = np.flatnonzero(routes == r)
            ctx = w if r == WINDOW else np.concatenate([self.archives[int(r)], w])
            groups.append((pos, ctx, int(r)))
        return groups

    def stored_rows(self) -> int:
        return self.win.stored_rows() + sum(len(a) for a in self.archives)

    def max_context(self) -> int:
        a = self.per_class * self.n_classes
        return self.win.window + (a * self.max_archives if self.mode == "union" else a)

    def digest(self) -> str:
        return _hash(self.win.window_idx(), *self.archives)


METHODS = {
    **{f"sw{w}": ("sliding", w) for w in (100, 200, 300, 400, 600, 1000, 1500, 2000)},
    "dual1000": ("dual", 1000),
    "dual400": ("dual", 400),
    "cbfifo400": ("cbfifo", 400),
    "arch_union": ("archive", "union"),
    "arch_routed": ("archive", "routed"),
    "arch_routed_adwin": ("archive", "routed"),
    "dual1000_rb": ("dual_rb", 1000),            # 修订 1：我们的方法（阶段均衡双记忆）
    "dual1000_rb_adwin": ("dual_rb", 1000),      # 同上，阶段由 ADWIN 报警划分
}


def make_memory(method: str, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int):
    """按预注册的方法名构造记忆。"""
    if method not in METHODS:
        raise ValueError(f"未知方法 {method!r}；可选 {sorted(METHODS)}")
    kind, arg = METHODS[method]
    if kind == "sliding":
        return SlidingMemory(arg)
    if kind == "dual":
        return DualMemory(arg, 0.75, y)
    if kind == "dual_rb":
        return RegimeBalancedDual(arg, 0.75, y)
    if kind == "cbfifo":
        return ClassBalancedFIFO(arg, y, n_classes)
    rng = np.random.default_rng(10_000 + seed)       # 档案抽样只依赖 seed
    return RegimeArchive(200, 25, arg, X, y, n_classes, rng, k=10, max_archives=8)

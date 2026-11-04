"""Round 2：决定每一步给 TabPFN 看哪些已揭示的行（"上下文记忆"）。

所有记忆只存**流内下标**，按时间升序；标签揭示后才通过 `update` 进入记忆，
所以任何上下文行都早于被预测的行（由 runner 再断言一次）。

统一接口：
    update(idx)                  新揭示的流内下标（升序）
    close_regime(rows)           一个阶段结束（档案类方法与阶段均衡双记忆 *_rb / *_rbr 用到，其余方法忽略；
                                 CBRS 双记忆 *_cbrs 也忽略）
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


class ReservoirDual:
    """CBRS 双记忆（预注册修订 4 的基线 2；Round 5）。

    结构与 `DualMemory` 完全相同：短期 FIFO（total × short_ratio 行）+ 长期池，短期溢出的最老一条移入长期池。
    长期池按 Chrysakis & Moens 2020（ICML，PMLR v119）"Online Continual Learning from Imbalanced Data"
    的 class-balancing reservoir sampling（CBRS，Algorithm 1）维护；"流" = 依次进入长期池的行。

    "满类"的定义（原文）："we will call a class full if it currently is, or has been in one of the previous
    time steps, the largest class. Once a class becomes full, it remains so in the future."
    实现：每次存入/替换之后（包括池未满的填充阶段），把当前行数最多的类（含并列）加入 `_full`，且永不移出。

    Algorithm 1（对每一条进入长期池的行 (x, c)；n_c 先加 1）：
      - 池未满：直接存入；
      - 池满且 c 不是满类：从当前行数最多的类（并列时均匀随机选一类）里均匀随机选一行，用新行覆盖；
      - 池满且 c 是满类：以概率 m_c / n_c 用新行覆盖池中均匀随机的一条 c 类行，否则丢弃新行。
        m_c = 池中 c 类行数，n_c = 迄今进入过长期池的 c 类行数（含这一行）。
    不使用任何阶段信息（`close_regime` 什么都不做）；随机数只依赖 seed（`default_rng(20_000 + seed)`）。
    池内行始终按时间升序（新行总是最新的，覆盖 = 删旧行 + 追加；上下文是集合，顺序不影响语义）。
    """

    def __init__(self, total: int, short_ratio: float, y: np.ndarray, seed: int = 0):
        self.short_cap = max(1, int(round(total * short_ratio)))
        self.long_cap = int(total) - self.short_cap
        assert self.long_cap >= 1
        self.y = y
        self._short: list = []
        self._long: list = []
        self._n_seen: dict = {}                       # 类别 -> 迄今进入过长期池的行数 n_c
        self._full: set = set()                       # 满类集合（单调增长）
        self._rng = np.random.default_rng(20_000 + int(seed))

    def update(self, idx) -> None:
        for i in idx:
            self._short.append(int(i))
            if len(self._short) > self.short_cap:
                self._to_long(self._short.pop(0))

    def _mark_full(self, ys: np.ndarray) -> None:
        classes, counts = np.unique(ys, return_counts=True)
        self._full.update(int(c) for c in classes[counts == counts.max()])

    def _to_long(self, i: int) -> None:
        c = int(self.y[i])
        self._n_seen[c] = self._n_seen.get(c, 0) + 1
        if len(self._long) < self.long_cap:            # 池未满：存入
            self._long.append(i)
            self._mark_full(np.asarray(self.y)[np.asarray(self._long, dtype=np.int64)])
            return
        ys = np.asarray(self.y)[np.asarray(self._long, dtype=np.int64)]
        if c not in self._full:                        # 非满类：覆盖当前最多的类里的一条
            classes, counts = np.unique(ys, return_counts=True)
            tied = classes[counts == counts.max()]
            victim_class = int(tied[self._rng.integers(len(tied))])
            pos_pool = np.flatnonzero(ys == victim_class)
        else:                                          # 满类：以 m_c / n_c 覆盖同类一条，否则丢弃
            pos_pool = np.flatnonzero(ys == c)
            if not self._rng.random() < len(pos_pool) / self._n_seen[c]:
                return
        self._long.pop(int(pos_pool[self._rng.integers(len(pos_pool))]))
        self._long.append(i)
        self._mark_full(np.asarray(self.y)[np.asarray(self._long, dtype=np.int64)])

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
        n = sorted(self._n_seen.items())
        return _hash(self._long, self._short, [k for k, _ in n], [v for _, v in n], sorted(self._full))


class RegimeReservoirDual:
    """阶段均衡的蓄水池双记忆（预注册修订 5；Round 6）。

    结构与 `DualMemory` / `RegimeBalancedDual` / `ReservoirDual` 相同：短期 FIFO（total × short_ratio 行）+ 长期池，
    短期溢出的最老一条进入长期池。长期池把 CBRS（Chrysakis & Moens 2020，Algorithm 1）用在两层：先把阶段当作"类"
    在阶段之间均衡，再在每个阶段内部按类别均衡；每个（阶段, 类别）格子内部是蓄水池抽样。
    与 `RegimeBalancedDual` 的区别只在阶段内部：那里淘汰最老的行（只剩每个阶段最后一段），这里均匀抽样。

    对每一条进入长期池的行 x（g = x 揭示时的阶段号，c = 类别；n[g,c] = 迄今进入过长期池的 (g, c) 行数，先加 1）：
      1. 池未满：存入；
      2. g 不是满阶段：在行数最多的阶段 g*（并列取较新的阶段）里找行数最多的类 c*（并列均匀随机），
         均匀随机删一条 (g*, c*) 的行，存入 x；
      3. g 是满阶段、c 不是 g 内的满类：在 g 内行数最多的类 c*（并列均匀随机）里均匀随机删一条，存入 x；
      4. 否则：以概率 m[g,c] / n[g,c] 用 x 覆盖池中均匀随机的一条 (g, c) 行（m = 池中 (g, c) 行数），否则丢弃 x。
    每次存入或覆盖之后（含填充阶段）：当前行数最多的阶段（含并列）加入满阶段集合；每个阶段内当前行数最多的类（含并列）
    加入该阶段的满类集合。两种集合都永不移出（CBRS 的"满类"定义用在两层）。
    只有一个阶段（从不调用 `close_regime`）时，决策与 `ReservoirDual` 完全相同（同一随机数序列下逐条一致）。
    阶段由 `close_regime` 划分（只用官方变点）；随机数只依赖 seed（`default_rng(30_000 + seed)`）。
    池内行始终按时间升序（覆盖 = 删旧行 + 追加新行）。
    """

    def __init__(self, total: int, short_ratio: float, y: np.ndarray, seed: int = 0, rng_base: int = 30_000):
        self.short_cap = max(1, int(round(total * short_ratio)))
        self.long_cap = int(total) - self.short_cap
        assert self.long_cap >= 1
        self.y = y
        self._short: list = []
        self._long: list = []
        self._seg: dict = {}                           # 行下标 -> 阶段号（揭示时记下）
        self._cur = 0
        self._n_seen: dict = {}                        # (阶段, 类别) -> 迄今进入过长期池的行数 n[g,c]
        self._full_r: set = set()                      # 满阶段集合（单调增长）
        self._full_c: dict = {}                        # 阶段 -> 该阶段内的满类集合（单调增长）
        self._rng = np.random.default_rng(int(rng_base) + int(seed))

    def update(self, idx) -> None:
        for i in idx:
            i = int(i)
            self._seg[i] = self._cur
            self._short.append(i)
            if len(self._short) > self.short_cap:
                self._to_long(self._short.pop(0))

    def _pool(self) -> "tuple[np.ndarray, np.ndarray]":
        long_ = np.asarray(self._long, dtype=np.int64)
        segs = np.fromiter((self._seg[i] for i in self._long), dtype=np.int64, count=len(self._long))
        return segs, np.asarray(self.y)[long_]

    def _mark_full(self) -> None:
        segs, ys = self._pool()
        gs, gc = np.unique(segs, return_counts=True)
        self._full_r.update(int(g) for g in gs[gc == gc.max()])
        for g in gs:
            classes, counts = np.unique(ys[segs == g], return_counts=True)
            self._full_c.setdefault(int(g), set()).update(int(c) for c in classes[counts == counts.max()])

    def _largest_class_rows(self, ys: np.ndarray, in_g: np.ndarray) -> np.ndarray:
        classes, counts = np.unique(ys[in_g], return_counts=True)
        tied = classes[counts == counts.max()]
        victim_class = int(tied[self._rng.integers(len(tied))])
        return np.flatnonzero(in_g & (ys == victim_class))

    def _to_long(self, i: int) -> None:
        g, c = self._seg[i], int(self.y[i])
        self._n_seen[(g, c)] = self._n_seen.get((g, c), 0) + 1
        if len(self._long) < self.long_cap:            # 1. 池未满：存入
            self._long.append(i)
            self._mark_full()
            return
        segs, ys = self._pool()
        if g not in self._full_r:                      # 2. 非满阶段：覆盖最大阶段里最大类的一条
            gs, gc = np.unique(segs, return_counts=True)
            g_star = int(gs[gc == gc.max()].max())
            pos_pool = self._largest_class_rows(ys, segs == g_star)
        elif c not in self._full_c.get(g, set()):     # 3. 阶段内的非满类：覆盖本阶段最大类的一条
            in_g = segs == g
            assert in_g.any(), "满阶段在池中没有行时又有新行进入（预注册修订 5 的网格里不会发生）"
            pos_pool = self._largest_class_rows(ys, in_g)
        else:                                          # 4. 满格子：以 m/n 覆盖同格子一条，否则丢弃
            pos_pool = np.flatnonzero((segs == g) & (ys == c))
            if not self._rng.random() < len(pos_pool) / self._n_seen[(g, c)]:
                del self._seg[i]
                return
        del self._seg[self._long.pop(int(pos_pool[self._rng.integers(len(pos_pool))]))]
        self._long.append(i)
        self._mark_full()

    def close_regime(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return
        cut = int(rows.max()) + 1
        self._cur += 1
        assert all(i < cut for i in self._long), "关闭点之后的行已进入长期池（短期容量小于一批？）"
        for i in self._short:                         # 关闭点之后、已揭示的行属于新阶段
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
        n = sorted(self._n_seen.items())
        fc = sorted((g, c) for g, cs in self._full_c.items() for c in cs)
        return _hash(self._long, self._short, [self._seg[i] for i in self._long + self._short], [self._cur],
                     [g for (g, _), _ in n], [c for (_, c), _ in n], [v for _, v in n], sorted(self._full_r),
                     [g for g, _ in fc], [c for _, c in fc])


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
    # 修订 3：200 行规模（短期 150 + 长期 50）
    "dual200": ("dual", 200),
    "dual200_rb": ("dual_rb", 200),
    "dual200_rb_adwin": ("dual_rb", 200),
    # 探索性比例分析（预注册修订 3 的补充说明，2026-10-17）：200 行，短期占比 0.65 / 0.85（KDD 论文的网格；
    # 0.75 就是上面的 dual200 / dual200_rb）。第三个元素是短期占比，缺省为 0.75
    **{f"dual200_s{int(r * 100)}": ("dual", 200, r) for r in (0.65, 0.85)},
    **{f"dual200_rb_s{int(r * 100)}": ("dual_rb", 200, r) for r in (0.65, 0.85)},
    # Round 5（预注册修订 4，审阅后补做）：
    # 基线 2：CBRS 双记忆（长期池满时用类均衡蓄水池抽样，不用阶段信息）
    "dual200_cbrs": ("dual_cbrs", 200),
    "dual1000_cbrs": ("dual_cbrs", 1000),
    # 消融：与 dual200 / dual1000 完全相同的记忆，但 run_round2.py 给每行特征末尾加一列官方阶段编号
    "dual200_rid": ("dual", 200),
    "dual1000_rid": ("dual", 1000),
    # Round 6（预注册修订 5）：阶段均衡的蓄水池双记忆（阶段之间均衡，阶段内部按类别均衡 + 蓄水池抽样）
    "dual200_rbr": ("dual_rbr", 200),
    "dual1000_rbr": ("dual_rbr", 1000),
}


def make_memory(method: str, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int):
    """按预注册的方法名构造记忆。"""
    if method not in METHODS:
        raise ValueError(f"未知方法 {method!r}；可选 {sorted(METHODS)}")
    kind, arg, *rest = METHODS[method]
    short_ratio = rest[0] if rest else 0.75
    if kind == "sliding":
        return SlidingMemory(arg)
    if kind == "dual":
        return DualMemory(arg, short_ratio, y)
    if kind == "dual_rb":
        return RegimeBalancedDual(arg, short_ratio, y)
    if kind == "dual_cbrs":
        return ReservoirDual(arg, short_ratio, y, seed)
    if kind == "dual_rbr":
        return RegimeReservoirDual(arg, short_ratio, y, seed)
    if kind == "cbfifo":
        return ClassBalancedFIFO(arg, y, n_classes)
    rng = np.random.default_rng(10_000 + seed)       # 档案抽样只依赖 seed
    return RegimeArchive(200, 25, arg, X, y, n_classes, rng, k=10, max_archives=8)

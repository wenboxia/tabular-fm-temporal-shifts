"""原子写 npz：先写临时文件，再 os.replace 成正式文件名。

multiseed 驱动靠"结果 npz 是否存在"判断一个运行是否已完成（skip-existing）。
进程在写文件的一瞬间被杀（关窗口、断电）会留下半截 npz，它会被当成已完成而永远跳过。
原子落盘保证：正式文件名下要么没有文件，要么是完整的文件。
"""
import os

import numpy as np


def atomic_savez(path: str, **arrays) -> str:
    """与 np.savez(path, **arrays) 等价，但不会留下半截文件。返回最终路径。"""
    if not path.endswith(".npz"):
        path += ".npz"
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path

"""通用小工具。"""
from __future__ import annotations

import sys


def disable_power_throttling() -> bool:
    """只对本进程关闭 Windows 11 的 EcoQoS 后台节流, 返回是否成功 (非 Windows 直接返回 False)。

    实测: 没有前台窗口的命令行进程跑约 2 秒后会被系统当成 "后台任务", 调到能效核/降频,
    MediaPipe 从 7 ms/帧 变成 24 ms/帧 (sleep 也恢复不了)。评测、训练、离线处理视频时关掉它,
    数字才和真实使用一致。这只是本进程的一个 QoS 提示, 进程退出即失效, 不改任何系统设置。
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    # ProcessPowerThrottling = 4; ControlMask = EXECUTION_SPEED(1); StateMask = 0 表示 "不节流"
    st = PowerThrottlingState(1, 0x1, 0x0)
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))

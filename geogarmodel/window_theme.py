"""Native Windows 11 dark caption, border and rounded corners."""
import sys


def apply_window_theme(widget):
    if sys.platform != "win32":
        return
    from PyQt5.QtWidgets import QApplication
    if QApplication.platformName() != "windows":
        return
    import ctypes
    from ctypes import wintypes
    try:
        setter = ctypes.WinDLL("dwmapi").DwmSetWindowAttribute
        setter.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        setter.restype = ctypes.c_long
        # COLORREF is 0x00BBGGRR. Unsupported attributes leave native defaults.
        results = {}
        for attribute, value in ((20, 1), (33, 2), (34, 0x004E3624),
                                 (35, 0x002F1C10), (36, 0x00F6E7DB)):
            data = wintypes.DWORD(value)
            results[attribute] = setter(int(widget.winId()), attribute, ctypes.byref(data), ctypes.sizeof(data))
        return results
    except (OSError, AttributeError):
        pass

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
        from .theme import current_theme
        t=current_theme()
        def colorref(value):
            r,g,b=(int(value[i:i+2],16) for i in (1,3,5))
            return r|(g<<8)|(b<<16)
        results = {}
        for attribute, value in ((20, int(not t.light)), (33, 2), (34, colorref(t.border)),
                                 (35, colorref(t.panel)), (36, colorref(t.text))):
            data = wintypes.DWORD(value)
            results[attribute] = setter(int(widget.winId()), attribute, ctypes.byref(data), ctypes.sizeof(data))
        return results
    except (OSError, AttributeError):
        pass

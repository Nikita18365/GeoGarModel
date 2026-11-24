"""Store an optional web-search credential in the current Windows profile."""
import base64
import ctypes
import os
import sys
from ctypes import wintypes


CAN_REMEMBER_KEY = sys.platform == 'win32'
SAVED_KEY = 'assistant/ollamaWebKeyEncrypted'
ENABLED = 'assistant/ollamaWebSearchEnabled'
DEVICE_SEARCH = 'assistant/ollamaDeviceSearch'


class _Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.c_void_p)]


def _protect(data, encrypt):
    api = ctypes.WinDLL('crypt32', use_last_error=True)
    transform = api.CryptProtectData if encrypt else api.CryptUnprotectData
    transform.argtypes = (ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob))
    transform.restype = wintypes.BOOL
    free = ctypes.WinDLL('kernel32').LocalFree
    free.argtypes = (ctypes.c_void_p,)
    free.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.addressof(buffer))
    result = _Blob()
    # CRYPTPROTECT_UI_FORBIDDEN. Encryption uses the current user's scope.
    if not transform(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        free(result.data)


def valid_web_key(key):
    return 0 < len(key) <= 2048 and all('!' <= char <= '~' for char in key)


class OllamaWebCredential:
    def __init__(self, settings):
        self.settings = settings
        self.session_key = ''

    @property
    def environment_key(self):
        key = os.environ.get('OLLAMA_API_KEY', '').strip()
        return key if valid_web_key(key) else ''

    def current(self):
        if not self.settings.value(ENABLED, True, type=bool):
            return ''
        if self.session_key:
            return self.session_key
        saved = self.settings.value(SAVED_KEY, '', type=str)
        if CAN_REMEMBER_KEY and saved:
            try:
                key = _protect(base64.b64decode(saved, validate=True), False).decode('ascii')
                if valid_web_key(key):
                    return key
            except (OSError, ValueError):
                pass
        return self.environment_key

    def connect(self, key, remember):
        self.session_key = key
        self.settings.setValue(ENABLED, True)
        self.settings.remove(SAVED_KEY)
        if remember and CAN_REMEMBER_KEY:
            encrypted = base64.b64encode(_protect(key.encode('ascii'), True)).decode('ascii')
            self.settings.setValue(SAVED_KEY, encrypted)
        self.settings.sync()
        return self.settings.status() == self.settings.NoError

    def disconnect(self):
        self.session_key = ''
        self.settings.setValue(ENABLED, False)
        self.settings.remove(SAVED_KEY)
        self.settings.sync()
        return self.settings.status() == self.settings.NoError

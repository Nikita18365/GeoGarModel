from urllib.error import HTTPError

import pytest
from PyQt5.QtCore import QSettings, QThreadPool
from PyQt5.QtWidgets import QDialog, QLineEdit

from geogarmodel import ollama_web_ui
from geogarmodel.ollama_credentials import CAN_REMEMBER_KEY, SAVED_KEY, OllamaWebCredential


@pytest.mark.skipif(not CAN_REMEMBER_KEY, reason='Windows DPAPI')
def test_windows_key_survives_reopen_in_encrypted_form(tmp_path, monkeypatch):
    path = str(tmp_path / 'ollama.ini')
    settings = QSettings(path, QSettings.IniFormat)
    credential = OllamaWebCredential(settings)
    monkeypatch.setenv('OLLAMA_API_KEY', 'environment-example')
    assert credential.current() == 'environment-example'
    assert credential.connect('window-example-token', remember=True)
    assert settings.value(SAVED_KEY)
    assert 'window-example-token' not in (tmp_path / 'ollama.ini').read_text()

    reopened = OllamaWebCredential(QSettings(path, QSettings.IniFormat))
    assert reopened.current() == 'window-example-token'
    assert reopened.disconnect()
    assert OllamaWebCredential(QSettings(path, QSettings.IniFormat)).current() == ''
    assert not reopened.settings.contains(SAVED_KEY)


@pytest.mark.parametrize('code, message', [(401, 'отклонила ключ'), (402, 'по условиям оплаты')])
def test_dialog_checks_key_before_connecting_and_allows_retry(tmp_path, qt_application_lifetime, monkeypatch,
                                                              code, message):
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
    settings = QSettings(str(tmp_path / 'ollama.ini'), QSettings.IniFormat)
    credential = OllamaWebCredential(settings)
    calls = []

    def verify(query, key):
        calls.append(query)
        if key != 'valid-example':
            raise HTTPError('https://ollama.com/api/web_search', code, 'rejected', {}, None)
        return ()

    monkeypatch.setattr(ollama_web_ui, 'search_ollama', verify)
    dialog = ollama_web_ui.OllamaWebDialog(None, credential)
    dialog.remember.setChecked(False)
    assert dialog.key.echoMode() == QLineEdit.Password
    try:
        dialog.key.setText('invalid-example')
        dialog.verify()
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
        assert message in dialog.status.text()
        assert dialog.billing_link.isHidden() == (code != 402)
        assert credential.current() == ''
        dialog.key.setText('valid-example')
        dialog.verify()
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
        assert dialog.result() == QDialog.Accepted
        assert credential.current() == 'valid-example'
        assert OllamaWebCredential(settings).current() == ''
        assert calls == ['geophysical forward modelling'] * 2
    finally:
        dialog.close()


def test_web_search_402_preserves_public_fallback(monkeypatch):
    from geogarmodel import manual_assistant

    def payment_required(*args):
        raise HTTPError('https://ollama.com/api/web_search', 402, 'payment required', {}, None)

    source = manual_assistant.Passage('Википедия', 'Гравиразведка', 'Описание', 'https://ru.wikipedia.org/')
    monkeypatch.setattr(manual_assistant, 'search_ollama', payment_required)
    monkeypatch.setattr(manual_assistant, 'search_web', lambda topic: (source,))
    sources, notice = manual_assistant.external_sources('Как рассчитывается сила тяжести?', {'грави'}, [],
                                                       web_key='test-key')
    assert sources == (source,) and 'HTTP 402' in notice and 'Использованы открытые источники' in notice

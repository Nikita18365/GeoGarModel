import pytest
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


@pytest.fixture(scope='session',autouse=True)
def qt_application_lifetime():
    # Qt supports one QApplication per process; keep it alive across test modules.
    from PyQt5.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def discard_unsaved_map_projects(monkeypatch):
    # Tests close their temporary windows without opening an interactive save prompt.
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.Discard)


@pytest.fixture(autouse=True)
def isolate_assistant_history(monkeypatch, tmp_path):
    from PyQt5.QtCore import QSettings
    from geogarmodel import manual_assistant_ui
    from geogarmodel.assistant_history import ChatStore

    monkeypatch.setattr(manual_assistant_ui, 'chat_store',
                        lambda: ChatStore(tmp_path / 'assistant.sqlite3'))
    monkeypatch.setattr(manual_assistant_ui, 'QSettings',
                        lambda *_: QSettings(str(tmp_path / 'assistant.ini'), QSettings.IniFormat))

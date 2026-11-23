"""Connection dialog for Ollama Web Search."""
from urllib.error import HTTPError, URLError

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout,
)

from .manual_assistant import OLLAMA_BILLING_REQUIRED, search_ollama
from .ollama_credentials import CAN_REMEMBER_KEY, valid_web_key


class KeyCheckSignal(QObject):
    done = pyqtSignal(str)


class KeyCheck(QRunnable):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.signal = KeyCheckSignal()

    def run(self):
        try:
            search_ollama('geophysical forward modelling', self.key)
        except HTTPError as error:
            message = {
                401: 'Ollama отклонила ключ. Проверьте, что он скопирован полностью.',
                402: OLLAMA_BILLING_REQUIRED,
                403: 'Ollama запретила доступ. Проверьте ключ и доступ к веб-поиску в аккаунте.',
                429: 'Превышен лимит запросов Ollama. Повторите проверку позже.',
            }.get(error.code, f'Ollama вернула ошибку HTTP {error.code}.')
        except (URLError, OSError):
            message = 'Не удалось подключиться к Ollama. Проверьте интернет и повторите попытку.'
        except (ValueError, KeyError, TypeError, AttributeError):
            message = 'Ollama вернула ответ в неожиданном формате.'
        else:
            message = ''
        finally:
            self.key = ''
        self.signal.done.emit(message)


class OllamaWebDialog(QDialog):
    def __init__(self, parent, credential):
        super().__init__(parent)
        self.credential = credential
        self.pending_key = ''
        self.result_message = ''
        self.setWindowTitle('Подключение Ollama Web Search')
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        help_link = QLabel('Войдите в аккаунт Ollama и '
                           '<a href="https://ollama.com/settings/keys">создайте ключ</a>. '
                           'Вставьте его в поле ниже. При интернет-поиске Ollama получает текущий вопрос.')
        help_link.setOpenExternalLinks(True)
        help_link.setWordWrap(True)
        layout.addWidget(help_link)
        current = credential.current()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setAccessibleName('Ключ Ollama API')
        self.key.setMaxLength(2048)
        self.key.setPlaceholderText('Вставьте ключ для замены' if current else 'Вставьте ключ Ollama API')
        layout.addWidget(self.key)
        self.remember = QCheckBox('Запомнить в профиле Windows')
        self.remember.setChecked(CAN_REMEMBER_KEY)
        if CAN_REMEMBER_KEY:
            layout.addWidget(self.remember)
        else:
            layout.addWidget(QLabel('Ключ из этого окна действует до закрытия программы.'))
        self.status = QLabel('Ключ подключён.' if current else 'Ключ не подключён.')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.billing_link = QLabel('<a href="https://ollama.com/settings/billing">Открыть настройки оплаты Ollama</a>')
        self.billing_link.setOpenExternalLinks(True)
        self.billing_link.hide()
        layout.addWidget(self.billing_link)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.verify_button = buttons.addButton('Проверить и подключить', QDialogButtonBox.ActionRole)
        self.disconnect_button = buttons.addButton('Отключить', QDialogButtonBox.ActionRole)
        self.disconnect_button.setEnabled(bool(current))
        buttons.rejected.connect(self.reject)
        self.verify_button.clicked.connect(self.verify)
        self.disconnect_button.clicked.connect(self.disconnect)
        self.key.returnPressed.connect(self.verify)
        layout.addWidget(buttons)

    def verify(self):
        if self.pending_key:
            return
        typed = self.key.text().strip()
        key = typed or self.credential.current() or self.credential.environment_key
        if not valid_web_key(key):
            self.status.setText('Вставьте ключ Ollama без пробелов.')
            return
        self.pending_key = key
        self.key.clear()
        self.key.setEnabled(False)
        self.remember.setEnabled(False)
        self.verify_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.billing_link.hide()
        self.status.setText('Проверяю ключ…')
        self.request = KeyCheck(key)
        self.request.signal.done.connect(self.checked)
        QThreadPool.globalInstance().start(self.request)

    def checked(self, error):
        key, self.pending_key = self.pending_key, ''
        if not key:
            return
        if error:
            self.status.setText(error)
            self.billing_link.setVisible(error == OLLAMA_BILLING_REQUIRED)
            self.key.setEnabled(True)
            self.remember.setEnabled(True)
            self.verify_button.setEnabled(True)
            self.disconnect_button.setEnabled(bool(self.credential.current()))
            self.key.setFocus()
            return
        try:
            saved = self.credential.connect(key, self.remember.isChecked())
        except OSError:
            saved = False
        self.result_message = ('Веб-поиск Ollama подключён.' if saved else
                               'Веб-поиск подключён на этот сеанс. Не удалось сохранить настройки.')
        self.accept()

    def disconnect(self):
        saved = self.credential.disconnect()
        self.result_message = ('Веб-поиск Ollama отключён. Доступны открытые источники.' if saved else
                               'Веб-поиск отключён на этот сеанс. Не удалось сохранить настройки.')
        self.accept()

    def reject(self):
        self.pending_key = ''
        self.key.clear()
        super().reject()

"""Connection dialog for Ollama Web Search."""
from pathlib import Path
from urllib.error import HTTPError, URLError

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from .manual_assistant import OLLAMA_BILLING_REQUIRED, search_ollama
from .ollama_credentials import CAN_REMEMBER_KEY, DEVICE_SEARCH, valid_web_key


class OllamaDeviceDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle('Подключение устройства Ollama')
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        instructions = QLabel(
            'Войдите в Ollama App или выполните <b>ollama signin</b> в терминале. '
            'Для ручной привязки скопируйте публичный Device Key и вставьте его '
            'в <a href="https://ollama.com/settings/keys">настройках аккаунта Ollama</a>.')
        instructions.setWordWrap(True)
        instructions.setOpenExternalLinks(True)
        layout.addWidget(instructions)
        copy = QPushButton('Скопировать публичный Device Key')
        copy.clicked.connect(self.copy_public_key)
        layout.addWidget(copy)
        self.use_device = QCheckBox('Использовать поиск через Ollama App')
        self.use_device.setChecked(settings.value(DEVICE_SEARCH, False, type=bool))
        layout.addWidget(self.use_device)
        self.status = QLabel(
            'Ollama App получает текущий вопрос и выполняет веб-поиск через ваш аккаунт. '
            'Доступ зависит от версии Ollama и условий аккаунта.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def copy_public_key(self):
        try:
            with (Path.home() / '.ollama' / 'id_ed25519.pub').open(encoding='ascii') as file:
                key = file.read(4096).strip()
        except (OSError, ValueError):
            key = ''
        fields = key.split()
        if len(fields) < 2 or fields[0] != 'ssh-ed25519':
            self.status.setText('Публичный ключ Ollama не найден. Запустите Ollama App и войдите в аккаунт.')
            return
        QApplication.clipboard().setText(' '.join(fields[:2]))
        self.status.setText('Публичный ключ скопирован. Вставьте его в настройках аккаунта Ollama.')

    def accept(self):
        self.settings.setValue(DEVICE_SEARCH, self.use_device.isChecked())
        self.settings.sync()
        super().accept()


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
        self.device_button = QPushButton('Подключение через Ollama App…')
        self.device_button.clicked.connect(self.configure_device)
        layout.addWidget(self.device_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.verify_button = buttons.addButton('Проверить и подключить', QDialogButtonBox.ActionRole)
        self.disconnect_button = buttons.addButton('Отключить', QDialogButtonBox.ActionRole)
        self.disconnect_button.setEnabled(bool(current))
        buttons.rejected.connect(self.reject)
        self.verify_button.clicked.connect(self.verify)
        self.disconnect_button.clicked.connect(self.disconnect)
        self.key.returnPressed.connect(self.verify)
        layout.addWidget(buttons)

    def configure_device(self):
        dialog = OllamaDeviceDialog(self, self.credential.settings)
        if dialog.exec_() == QDialog.Accepted:
            self.result_message = ('Поиск через Ollama App включён.' if dialog.use_device.isChecked()
                                   else 'Поиск через Ollama App отключён.')
            self.accept()
        dialog.deleteLater()

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
        self.device_button.setEnabled(False)
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
            self.device_button.setEnabled(True)
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

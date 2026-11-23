"""Select an Ollama model and show its exact setup command."""
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPlainTextEdit, QPushButton)

from .ollama_catalog import FEATURED, LIBRARY, catalog_models, installed_models, is_cloud, setup_commands, valid_model


class OllamaModels(QDialog):
    def __init__(self, window, current):
        super().__init__(window)
        self.activity = window.activity
        self.setWindowTitle('Модель помощника · Ollama')
        self.resize(600,320)
        layout = QFormLayout(self)
        self.models = QComboBox()
        self.models.setEditable(True)
        self.models.setInsertPolicy(QComboBox.NoInsert)
        self.models.lineEdit().setPlaceholderText('например, qwen3.5:4b')
        self.models.addItems(list(dict.fromkeys((*FEATURED,current))))
        self.models.setCurrentText(current)
        layout.addRow('Модель или тег',self.models)

        actions = QHBoxLayout()
        layout.addRow(actions)
        refresh = QPushButton('Обновить установленные')
        refresh.clicked.connect(self.refresh_installed)
        actions.addWidget(refresh)
        catalog = QPushButton('Выбрать из каталога…')
        catalog.clicked.connect(self.choose_catalog)
        actions.addWidget(catalog)

        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addRow(self.note)
        self.commands = QPlainTextEdit()
        self.commands.setReadOnly(True)
        self.commands.setMaximumHeight(72)
        layout.addRow('В PowerShell',self.commands)
        copy = QPushButton('Копировать команды')
        copy.clicked.connect(lambda:QApplication.clipboard().setText(self.commands.toPlainText()))
        layout.addRow(copy)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addRow(self.status)
        library = QPushButton('Каталог Ollama в браузере')
        library.clicked.connect(lambda:QDesktopServices.openUrl(QUrl(LIBRARY)))
        layout.addRow(library)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        self.models.currentTextChanged.connect(self.update_setup)
        self.update_setup()

    @property
    def selected(self):
        return self.models.currentText().strip()

    def update_setup(self,*args):
        model=self.selected
        self.commands.setPlainText(setup_commands(model))
        self.ok.setEnabled(valid_model(model))
        if is_cloud(model):
            self.note.setText('Облачная модель. Войдите в Ollama и запустите модель один раз. '
                              'После запуска можно выйти из терминала командой /bye. '
                              'Вопрос и выдержки отправляются в Ollama Cloud.')
        else:
            self.note.setText('Запустите Ollama и скачайте модель один раз. '
                              'Затем нажмите OK и задайте вопрос в чате.')

    def add_names(self,names):
        for name in names:
            if self.models.findText(name)<0:
                self.models.addItem(name)

    def refresh_installed(self):
        try:
            names=self.activity.run('Получение списка Ollama',installed_models)
        except (OSError,ValueError,KeyError,TypeError):
            self.status.setText('Ollama недоступна. Запустите её и нажмите «Обновить установленные».')
            return
        self.add_names(names)
        self.status.setText(f'Установленных моделей: {len(names)}.')

    def choose_catalog(self):
        try:
            families=self.activity.run('Открытие каталога Ollama',catalog_models)
            family,ok=QInputDialog.getItem(self,'Каталог Ollama','Семейство моделей',families,0,False)
            if not ok:
                return
            names=self.activity.run('Чтение тегов Ollama',catalog_models,family)
            name,ok=QInputDialog.getItem(self,'Варианты Ollama','Выберите точный тег',names,0,False)
            if ok:
                self.add_names([name])
                self.models.setCurrentText(name)
        except (OSError,ValueError):
            self.status.setText('Каталог недоступен. Откройте каталог в браузере и введите точный тег вручную.')

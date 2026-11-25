"""Desktop help chat with a local CPU/GPU switch."""
from html import escape
from pathlib import Path
import re
import sqlite3

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, QSettings, QStandardPaths, QUrl
from PyQt5.QtGui import (QColor, QDesktopServices, QFontDatabase, QTextCursor,
    QTextDocument, QTextDocumentFragment, QTextLength, QTextOption, QTextTableFormat)
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDialog, QHBoxLayout, QInputDialog,
    QLabel, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QTextBrowser, QTextEdit,
    QToolButton, QVBoxLayout, QWidget)

from .assistant_history import ChatStore, Message, conversation_context
from .manual_assistant import FREE_CHAT, HYBRID, INTERNET, LOCAL, SOURCE_MODES, OFF_TOPIC, NO_ANSWER, answer_question
from .ollama_catalog import DEFAULT_MODEL, is_cloud, valid_model
from .ollama_credentials import DEVICE_SEARCH, OllamaWebCredential


USER_MESSAGE_COLOR = '#203b54'
CONTINUED_STEPS = re.compile(r'(?m)^(?!1\. )\d+\. [^\n]*(?:\n\d+\. [^\n]*)+')


def chat_font():
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    if 'OpenAI Sans' in QFontDatabase().families():
        font.setFamily('OpenAI Sans')
    elif 'Inter' not in QFontDatabase().families():
        directory = Path(__file__).with_name('assets')
        for path in directory.glob('Inter-*.ttf'):
            QFontDatabase.addApplicationFont(str(path))
    if font.family()!='OpenAI Sans' and 'Inter' in QFontDatabase().families():
        font.setFamily('Inter')
    font.setPixelSize(15)
    return font


class ChatMarkdown(QTextDocument):
    def loadResource(self,resource_type,url):
        # Responses and external excerpts cannot ask Qt to fetch resources.
        return None

    def __init__(self,text,font):
        super().__init__()
        self.setDefaultFont(font)
        # Qt 5 renumbers ordered lists from one. Keep step numbers in excerpts.
        text = CONTINUED_STEPS.sub(
            lambda match: '  \n'.join(line.replace('. ',r'\. ',1) for line in match[0].splitlines()),
            text)
        features=QTextDocument.MarkdownFeatures(QTextDocument.MarkdownDialectGitHub|QTextDocument.MarkdownNoHTML)
        self.setMarkdown(text,features)
        changes = []
        block = self.begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                fmt = fragment.charFormat()
                if fmt.isAnchor() or fmt.isImageFormat():
                    changes.append((fragment.position(),fragment.length(),fmt))
                fragments += 1
            block = block.next()
        for start,length,fmt in reversed(changes):
            cursor = QTextCursor(self)
            cursor.setPosition(start)
            cursor.setPosition(start+length,QTextCursor.KeepAnchor)
            if fmt.isImageFormat():
                cursor.removeSelectedText()
            else:
                fmt.setAnchor(False)
                fmt.setAnchorHref('')
                cursor.setCharFormat(fmt)


class ChatInput(QPlainTextEdit):
    submitted = pyqtSignal()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key_Return,Qt.Key_Enter) and event.modifiers()==Qt.NoModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ReplySignal(QObject):
    done = pyqtSignal(object)


class ManualRequest(QRunnable):
    def __init__(self, question, device, mode, model, context, web_key, device_search=False):
        super().__init__()
        self.args = question, device, mode, model, context
        self.web_key = web_key
        self.device_search = device_search
        self.signal = ReplySignal()

    def run(self):
        try:
            answer = answer_question(*self.args, web_key=self.web_key, device_search=self.device_search)
        finally:
            self.web_key = ''
        self.signal.done.emit(answer)


def chat_store():
    folder = Path(QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation))
    return ChatStore(folder / 'GeoGarModel' / 'assistant.sqlite3')


class ManualChat(QDockWidget):
    def __init__(self, window, device):
        super().__init__('Помощник GeoGarModel',window)
        self.setObjectName('manualAssistant')
        self.device = device
        self.settings = QSettings('GeoGarModel','GeoGarModel')
        self.web_credential = OllamaWebCredential(self.settings)
        configured_model = self.settings.value('assistant/model',DEFAULT_MODEL,type=str)
        self.model = configured_model if valid_model(configured_model) else DEFAULT_MODEL
        try:
            self.store = chat_store()
        except (OSError, sqlite3.Error):
            self.store = ChatStore(None)
        self.pending = None
        self.sources = []
        font=chat_font()
        self.chat_font=font
        body = QWidget()
        column = QVBoxLayout(body)
        toolbar = QHBoxLayout()
        column.addLayout(toolbar)
        self.chats = QComboBox()
        self.chats.setMinimumContentsLength(14)
        self.chats.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.chats.setAccessibleName('История чатов')
        self.chats.currentIndexChanged.connect(self.load_chat)
        toolbar.addWidget(self.chats, 1)
        new_button = QPushButton('+ Новый чат')
        new_button.clicked.connect(self.new_chat)
        toolbar.addWidget(new_button)
        actions = QToolButton()
        actions.setText('⋯')
        actions.setAccessibleName('Действия с чатом')
        actions.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(actions)
        menu.addAction('Переименовать чат…', self.rename_chat)
        menu.addAction('Удалить чат…', self.delete_chat)
        actions.setMenu(menu)
        toolbar.addWidget(actions)
        self.history = QTextBrowser()
        self.history.setFont(font)
        self.history.document().setDefaultFont(font)
        self.history.document().setDocumentMargin(12)
        self.history.setLineWrapMode(QTextEdit.WidgetWidth)
        self.history.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history.setOpenLinks(False)
        self.history.anchorClicked.connect(self.show_source)
        self.history.setMinimumWidth(340)
        column.addWidget(self.history,1)
        self.model_button = QPushButton()
        self.model_button.clicked.connect(self.choose_model)
        column.addWidget(self.model_button)
        row = QHBoxLayout()
        column.addLayout(row)
        self.input = ChatInput()
        self.input.setFont(font)
        self.input.setStyleSheet('font-size: 15px;')
        self.input.setFixedHeight(self.input.fontMetrics().lineSpacing()*3+20)
        self.input.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.input.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input.setPlaceholderText('Вопрос по программе')
        self.input.submitted.connect(self.ask)
        row.addWidget(self.input,1)
        self.send = QPushButton('Спросить')
        self.send.clicked.connect(self.ask)
        row.addWidget(self.send,0,Qt.AlignBottom)
        self.mode = QComboBox()
        self.mode.setAccessibleName('Режим ответа')
        for key, label in SOURCE_MODES.items():
            self.mode.addItem(label, key)
        default_mode = HYBRID if self.settings.value('assistant/web', True, type=bool) else LOCAL
        saved_mode = self.settings.value('assistant/sourceMode', default_mode, type=str)
        self.mode.setCurrentIndex(self.mode.findData(saved_mode if saved_mode in SOURCE_MODES else default_mode))
        self.mode.setToolTip('Википедия получает название темы или ключевые слова. Если настроен ключ поиска Ollama, '
                             'его сервис получает текущий вопрос. Облачная модель получает также контекст чата.')
        self.mode.setItemData(self.mode.findData(FREE_CHAT),
                              'Свободный чат с локальной моделью без веб-поиска.', Qt.ToolTipRole)
        self.mode.currentIndexChanged.connect(self.update_mode)
        sources_row = QHBoxLayout()
        sources_row.addWidget(self.mode, 1)
        self.web_button = QPushButton()
        self.web_button.setAccessibleName('Настроить ключ веб-поиска Ollama')
        self.web_button.clicked.connect(self.configure_web_search)
        sources_row.addWidget(self.web_button)
        column.addLayout(sources_row)
        self.update_web_label()
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        if self.store.path is None:
            warning = QLabel('Хранилище чатов недоступно. Переписка сохранится только до закрытия программы.')
            warning.setWordWrap(True)
            column.addWidget(warning)
        self.setWidget(body)
        window.addDockWidget(Qt.RightDockWidgetArea,self)
        self.update_mode()
        self.refresh_chats(self.settings.value('assistant/activeChat', 0, type=int))
        self.hide()

    def update_web_label(self):
        connected = bool(self.web_credential.current())
        device_search = self.settings.value(DEVICE_SEARCH, False, type=bool)
        if device_search and not connected:
            self.web_button.setText('Поиск через Ollama App')
            self.web_button.setToolTip('Настроить поиск через подключённую Ollama App.')
            return
        self.web_button.setText('Веб-поиск ✓' if connected else 'Подключить поиск…')
        self.web_button.setToolTip('Ключ Ollama подключён. Нажмите для настройки.' if connected else
                                   'Подключить ключ Ollama для поиска по сайтам.')

    def configure_web_search(self):
        from .ollama_web_ui import OllamaWebDialog

        dialog = OllamaWebDialog(self, self.web_credential)
        try:
            if dialog.exec_() == QDialog.Accepted:
                self.update_web_label()
                self.status.setText(dialog.result_message)
        finally:
            dialog.deleteLater()

    def refresh_chats(self, selected=None):
        rows = self.store.chats()
        if not rows:
            selected = self.store.create()
            rows = self.store.chats()
        self.chats.blockSignals(True)
        self.chats.clear()
        for chat_id, title in rows:
            self.chats.addItem(title, chat_id)
        self.chats.setCurrentIndex(max(0, self.chats.findData(selected)))
        self.chats.blockSignals(False)
        self.load_chat()

    def load_chat(self):
        chat_id = self.chats.currentData()
        if self.store.path is not None:
            self.settings.setValue('assistant/activeChat', chat_id)
        self.chats.setToolTip(self.chats.currentText())
        self.history.clear()
        self.sources.clear()
        messages = self.store.messages(chat_id)
        if not messages:
            self.history.append('Задайте любой вопрос локальной модели.' if self.mode.currentData() == FREE_CHAT else
                                'Задайте вопрос о GeoGarModel или о методе, который используется в программе.')
        for message in messages:
            self.render_message(message)
        self.status.setText('Готовлю ответ…' if self.pending and self.pending[0] == chat_id else '')

    def new_chat(self):
        self.input.clear()
        self.refresh_chats(self.store.create())
        self.input.setFocus()

    def rename_chat(self):
        title, accepted = QInputDialog.getText(self, 'Переименовать чат', 'Название',
                                               text=self.chats.currentText())
        if accepted and title.strip():
            title = title.strip()[:120]
            self.store.rename(self.chats.currentData(), title)
            self.chats.setItemText(self.chats.currentIndex(), title)
            self.chats.setToolTip(title)

    def delete_chat(self):
        chat_id = self.chats.currentData()
        title = self.chats.currentText()
        if QMessageBox.question(self, 'Удалить чат', f'Удалить «{title}» и всю его переписку?',
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.store.delete(chat_id)
            self.input.clear()
            self.refresh_chats()

    def update_mode(self):
        mode = self.mode.currentData()
        self.settings.setValue('assistant/sourceMode', mode)
        self.web_button.setVisible(mode in (INTERNET, HYBRID))
        self.input.setPlaceholderText('Задайте любой вопрос' if mode == FREE_CHAT else 'Вопрос по программе')
        self.update_model_label()
        if self.chats.count() and not self.store.messages(self.chats.currentData()):
            self.load_chat()

    def active_model(self):
        if self.mode.currentData() != FREE_CHAT:
            return self.model
        default = DEFAULT_MODEL if is_cloud(self.model) else self.model
        model = self.settings.value('assistant/freeModel', default, type=str)
        return model if valid_model(model) and not is_cloud(model) else default

    def update_model_label(self):
        model = self.active_model()
        cloud = is_cloud(model)
        self.model_button.setText(f'Модель · {model}'+(' · облако' if cloud else ''))
        self.device.setEnabled(not cloud)

    def choose_model(self):
        from .ollama_models_ui import OllamaModels
        local_only = self.mode.currentData() == FREE_CHAT
        dialog=OllamaModels(self.parentWidget(),self.active_model(),local_only=local_only)
        dialog.refresh_installed()
        if dialog.exec_()==QDialog.Accepted:
            if local_only:
                self.settings.setValue('assistant/freeModel', dialog.selected)
            else:
                self.model=dialog.selected
                self.settings.setValue('assistant/model',self.model)
            self.update_model_label()

    def ask(self):
        question=self.input.toPlainText().strip()
        if not question or not self.send.isEnabled():
            return
        if len(question)>1000:
            self.status.setText('Сократите вопрос до 1000 символов.')
            return
        self.status.clear()
        self.input.clear()
        chat_id = self.chats.currentData()
        mode = self.mode.currentData()
        messages = self.store.messages(chat_id)
        context = conversation_context(messages, question, mode)
        message = Message('user', question, mode)
        self.store.append(chat_id, message)
        self.render_message(message)
        if not messages and self.chats.currentText() == 'Новый чат':
            title = ' '.join(question.split())
            if len(title) > 56:
                title = title[:55].rstrip() + '…'
            self.store.rename(chat_id, title)
            self.chats.setItemText(self.chats.currentIndex(), title)
            self.chats.setToolTip(title)
        self.send.setEnabled(False)
        self.status.setText('Готовлю ответ…' if mode == FREE_CHAT else 'Готовлю ответ по источникам…')
        web = mode in (INTERNET, HYBRID)
        web_key = self.web_credential.current() if web else ''
        device_search = web and self.settings.value(DEVICE_SEARCH, False, type=bool)
        self.request=ManualRequest(question,self.device.currentText(),mode,self.active_model(),context,web_key,device_search)
        self.request.signal.done.connect(self.received)
        self.pending=(chat_id, mode)
        QThreadPool.globalInstance().start(self.request)

    def render_message(self, message):
        if message.role == 'user':
            self.append_question(message.text)
            return
        if message.role == 'reference':
            return
        if message.text:
            self.append_markdown(message.text)
        if message.excerpt:
            for passage in message.sources:
                self.append_source(passage, excerpt=True)

    def append_question(self,question):
        cursor=QTextCursor(self.history.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertBlock()
        frame=QTextTableFormat()
        frame.setBorder(0)
        frame.setCellPadding(9)
        frame.setCellSpacing(0)
        frame.setTopMargin(6)
        frame.setBottomMargin(6)
        frame.setWidth(QTextLength(QTextLength.PercentageLength,100))
        frame.setColumnWidthConstraints([QTextLength(QTextLength.PercentageLength,width) for width in (13,87)])
        cell=cursor.insertTable(1,2,frame).cellAt(0,1)
        style=cell.format()
        from .theme import current_theme,theme_colors
        t=current_theme()
        style.setBackground(QColor(USER_MESSAGE_COLOR if t.name=='Midnight Blue' else theme_colors(t)['selection']))
        cell.setFormat(style)
        cell.firstCursorPosition().insertText(question)
        self.history.moveCursor(QTextCursor.End)
        self.history.ensureCursorVisible()

    def received(self,answer):
        chat_id, mode = self.pending
        self.pending = None
        self.send.setEnabled(True)
        if self.chats.findData(chat_id) < 0:
            return
        text = answer.text if answer.sources or mode == FREE_CHAT else ''
        if answer.outside:
            text = OFF_TOPIC
        elif mode != FREE_CHAT and not text and not (answer.excerpt and answer.sources):
            text = ('По найденным веб-источникам ответ подтвердить не удалось.'
                    if mode == INTERNET else NO_ANSWER)
        if answer.notice:
            self.save_reply(chat_id, Message('notice', answer.notice, mode))
            if mode in (INTERNET, FREE_CHAT) and not answer.sources and not text:
                if self.chats.currentData() == chat_id:
                    self.status.clear()
                return
        self.save_reply(chat_id, Message('assistant', text, mode, answer.sources, answer.excerpt))
        if self.chats.currentData() == chat_id:
            self.status.clear()

    def save_reply(self, chat_id, message):
        self.store.append(chat_id, message)
        if self.chats.currentData() == chat_id:
            self.render_message(message)

    def append_source(self,passage,excerpt=False):
        self.sources.append(passage)
        source=f'<a href="help:{len(self.sources)-1}">{escape(passage.file)} · {escape(passage.heading)}</a>'
        self.history.append(source)
        if excerpt:
            self.append_markdown(passage.text)

    def append_markdown(self,text):
        document=ChatMarkdown(text,self.chat_font)
        cursor=QTextCursor(self.history.document())
        cursor.movePosition(QTextCursor.End)
        cursor.insertBlock()
        cursor.insertFragment(QTextDocumentFragment(document))
        self.history.moveCursor(QTextCursor.End)
        self.history.ensureCursorVisible()

    def show_source(self,url):
        if url.scheme()!='help' or not url.path().isascii() or not url.path().isdigit():
            return
        index=int(url.path())
        if index>=len(self.sources):
            return
        passage=self.sources[index]
        if passage.url:
            QDesktopServices.openUrl(QUrl(passage.url))
            return
        from .manual_assistant import passages
        line='\n\n'.join(p.text for p in passages() if (p.file,p.heading)==(passage.file,passage.heading))
        dialog=QDialog(self)
        dialog.setWindowTitle(passage.file+' · '+passage.heading)
        dialog.resize(750,500)
        layout=QVBoxLayout(dialog)
        view=QTextBrowser()
        view.setOpenLinks(False)
        view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        document=ChatMarkdown(line,self.chat_font)
        document.setParent(view)
        view.setDocument(document)
        layout.addWidget(view)
        dialog.exec_()


def add_manual_assistant(window, header):
    from .assistant_pet import AssistantPet

    settings=QSettings('GeoGarModel','GeoGarModel')
    device=QComboBox()
    device.setAccessibleName('Устройство справочного помощника')
    device.addItems(['CPU','GPU'])
    device.setCurrentText(settings.value('assistant/device','CPU',type=str))
    device.setToolTip('Устройство справочного помощника. GPU позволяет Ollama использовать доступную видеопамять.')
    device.currentTextChanged.connect(lambda value:settings.setValue('assistant/device',value))
    header.addWidget(device)
    button=QPushButton('🦊 Справка')
    button.setToolTip('Помощник по GeoGarModel')
    header.addWidget(button)
    chat=ManualChat(window,device)
    def toggle():
        chat.setVisible(chat.isHidden())
        if not chat.isHidden():
            chat.input.setFocus()
    button.clicked.connect(toggle)
    chat.pet = AssistantPet(window,settings,toggle)
    pet_menu = QToolButton()
    pet_menu.setText('Питомец')
    pet_menu.setToolTip('Выбрать или показать питомца')
    pet_menu.setMenu(chat.pet.menu)
    pet_menu.setPopupMode(QToolButton.InstantPopup)
    header.addWidget(pet_menu)
    return chat

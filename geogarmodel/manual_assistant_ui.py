"""Desktop help chat with a local CPU/GPU switch."""
from html import escape
from pathlib import Path
import re

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, QSettings, QUrl
from PyQt5.QtGui import (QColor, QDesktopServices, QFontDatabase, QTextCursor,
    QTextDocument, QTextDocumentFragment, QTextLength, QTextOption, QTextTableFormat)
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QTextBrowser, QTextEdit, QToolButton, QVBoxLayout, QWidget)

from .manual_assistant import choose_passages, find_passages, verified_answer, OFF_TOPIC, NO_ANSWER, terms


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
    def __init__(self, question, sources, device, previous, use_web):
        super().__init__()
        self.args = question,sources,device,previous,use_web
        self.signal = ReplySignal()

    def run(self):
        self.signal.done.emit(choose_passages(*self.args))


class ManualChat(QDockWidget):
    def __init__(self, window, device):
        super().__init__('Помощник GeoGarModel',window)
        self.setObjectName('manualAssistant')
        self.device = device
        self.previous = ''
        self.sources = []
        font=chat_font()
        self.chat_font=font
        body = QWidget()
        column = QVBoxLayout(body)
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
        self.history.append('Задайте вопрос о GeoGarModel или о методе, который используется в программе.')
        column.addWidget(self.history,1)
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
        settings = QSettings('GeoGarModel','GeoGarModel')
        self.web = QCheckBox('Дополнять ответ интернет-источниками')
        self.web.setChecked(settings.value('assistant/web',True,type=bool))
        self.web.setToolTip('Поиск по теме в Википедии. В поиск отправляется название темы, а текст вопроса остаётся в локальной Ollama.')
        self.web.toggled.connect(lambda value:settings.setValue('assistant/web',value))
        column.addWidget(self.web)
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        self.setWidget(body)
        window.addDockWidget(Qt.RightDockWidgetArea,self)
        self.hide()

    def ask(self):
        question=self.input.toPlainText().strip()
        if not question or not self.send.isEnabled():
            return
        if len(question)>1000:
            self.status.setText('Сократите вопрос до 1000 символов.')
            return
        self.status.clear()
        self.input.clear()
        self.append_question(question)
        previous=self.previous
        answer=verified_answer(question)
        if answer is not None:
            self.pending=(question,answer.sources)
            self.received(answer)
            return
        candidates=find_passages(question,previous)
        if not candidates:
            in_scope='geogarmodel' in question.casefold() or 'програм' in question.casefold()
            self.history.append(NO_ANSWER if in_scope else OFF_TOPIC)
            return
        self.send.setEnabled(False)
        self.status.setText('Готовлю ответ по источникам…')
        self.request=ManualRequest(question,candidates,self.device.currentText(),previous,self.web.isChecked())
        self.request.signal.done.connect(self.received)
        self.pending=(question,candidates)
        QThreadPool.globalInstance().start(self.request)

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
        style.setBackground(QColor(USER_MESSAGE_COLOR))
        cell.setFormat(style)
        cell.firstCursorPosition().insertText(question)
        self.history.moveCursor(QTextCursor.End)
        self.history.ensureCursorVisible()

    def received(self,answer):
        question,candidates=self.pending
        self.send.setEnabled(True)
        self.status.clear()
        if answer is None:
            self.status.setText('Ollama недоступна. Поиск по руководству работает.')
            if terms(question):
                self.previous=question
            for passage in candidates[:1 if 'sigma' in terms(question) else 2]:
                self.append_source(passage,excerpt=True)
            return
        if answer.outside:
            self.history.append(OFF_TOPIC)
            return
        if not answer.text or not answer.sources:
            self.history.append(NO_ANSWER)
            if answer.sources:
                for passage in answer.sources[:2]:
                    self.append_source(passage,excerpt=True)
            return
        if terms(question):
            self.previous=question
        self.append_markdown(answer.text)
        for passage in answer.sources:
            self.append_source(passage)

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

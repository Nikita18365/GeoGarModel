"""Desktop help chat with a local CPU/GPU switch."""
from html import escape

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, QSettings
from PyQt5.QtGui import QColor, QFontDatabase, QTextCursor, QTextLength, QTextOption, QTextTableFormat
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from .manual_assistant import choose_passages, find_passages, OFF_TOPIC, NO_ANSWER, terms


USER_MESSAGE_COLOR = '#203b54'


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
    def __init__(self, question, sources, device, previous):
        super().__init__()
        self.args = question,sources,device,previous
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
        font=QFontDatabase.systemFont(QFontDatabase.GeneralFont)
        if 'OpenAI Sans' in QFontDatabase().families():
            font.setFamily('OpenAI Sans')
        font.setPixelSize(15)
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
        self.history.append('Задайте вопрос по документации GeoGarModel.')
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
        candidates=find_passages(question,previous)
        if not candidates:
            in_scope='geogarmodel' in question.casefold() or 'програм' in question.casefold()
            self.history.append(NO_ANSWER if in_scope else OFF_TOPIC)
            return
        self.send.setEnabled(False)
        self.status.setText('Ищу ответ в руководстве…')
        self.request=ManualRequest(question,candidates,self.device.currentText(),previous)
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

    def received(self,selected):
        question,candidates=self.pending
        self.send.setEnabled(True)
        self.status.clear()
        if selected is None:
            self.status.setText('Ollama недоступна. Поиск по руководству работает.')
            selected=candidates[:2]
        if selected==OFF_TOPIC:
            self.history.append(OFF_TOPIC)
            return
        if not selected:
            self.history.append(NO_ANSWER)
            return
        if terms(question):
            self.previous=question
        for passage in selected:
            self.sources.append(passage)
            source=f'<a href="help:{len(self.sources)-1}">{escape(passage.file)} · {escape(passage.heading)}</a>'
            self.history.append(source+'<br>'+escape(passage.text).replace('\n','<br>'))

    def show_source(self,url):
        if url.scheme()!='help':
            return
        passage=self.sources[int(url.path())]
        from .manual_assistant import passages
        line='\n\n'.join(p.text for p in passages() if (p.file,p.heading)==(passage.file,passage.heading))
        dialog=QDialog(self)
        dialog.setWindowTitle(passage.file+' · '+passage.heading)
        dialog.resize(750,500)
        layout=QVBoxLayout(dialog)
        view=QTextBrowser()
        view.document().setDefaultFont(self.chat_font)
        view.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        view.setPlainText(line)
        layout.addWidget(view)
        dialog.exec_()


def add_manual_assistant(window, header):
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
    return chat

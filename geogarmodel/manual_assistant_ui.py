"""Desktop help chat with a local CPU/GPU switch."""
from html import escape

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, QSettings
from PyQt5.QtWidgets import (QComboBox, QDockWidget, QDialog, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from .manual_assistant import choose_passages, find_passages, OFF_TOPIC, NO_ANSWER, terms


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
        body = QWidget()
        column = QVBoxLayout(body)
        self.history = QTextBrowser()
        self.history.setOpenLinks(False)
        self.history.anchorClicked.connect(self.show_source)
        self.history.setMinimumWidth(340)
        self.history.append('Задайте вопрос по документации GeoGarModel.')
        column.addWidget(self.history,1)
        row = QHBoxLayout()
        column.addLayout(row)
        self.input = QLineEdit()
        self.input.setMaxLength(1000)
        self.input.setPlaceholderText('Вопрос по программе')
        self.input.returnPressed.connect(self.ask)
        row.addWidget(self.input,1)
        self.send = QPushButton('Спросить')
        self.send.clicked.connect(self.ask)
        row.addWidget(self.send)
        self.status = QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        self.setWidget(body)
        window.addDockWidget(Qt.RightDockWidgetArea,self)
        self.hide()

    def ask(self):
        question=self.input.text().strip()
        if not question or not self.send.isEnabled():
            return
        self.input.clear()
        self.history.append('<b>Вы</b> · '+escape(question))
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

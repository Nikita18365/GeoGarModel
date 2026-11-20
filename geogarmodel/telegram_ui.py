"""Telegram connection and event preferences."""
import secrets
import sys
from PyQt5 import sip
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QFormLayout,QLabel,QLineEdit,
    QPushButton,QCheckBox,QDialogButtonBox,QMessageBox,QScrollArea,QWidget)
from .telegram_notifications import api_call,find_chat


class TelegramDialog(QDialog):
    def __init__(self,service,parent):
        super().__init__(parent); self.service=service; self.code=secrets.token_hex(8)
        self.setWindowTitle('Уведомления Telegram'); self.resize(560,690)
        root=QVBoxLayout(self); scroll=QScrollArea(); scroll.setWidgetResizable(True)
        body=QWidget(); layout=QVBoxLayout(body); scroll.setWidget(body); root.addWidget(scroll)
        note=QLabel('1. Создайте личного бота через @BotFather командой /newbot.\n'
                    '2. Вставьте токен ниже.\n3. Отправьте своему боту команду из поля «Код подключения».\n'
                    '4. Нажмите «Найти мой чат», проверьте тест и сохраните настройки.')
        note.setWordWrap(True); layout.addWidget(note)
        form=QFormLayout(); form.setRowWrapPolicy(QFormLayout.WrapAllRows); layout.addLayout(form)
        self.token=QLineEdit(); self.token.setEchoMode(QLineEdit.Password)
        self.token.setText(service.store.get()); form.addRow('Токен бота',self.token)
        self.command=QLineEdit(f'/start {self.code}'); self.command.setReadOnly(True)
        form.addRow('Код подключения — скопируйте целиком',self.command)
        self.chat=QLineEdit(str(service.settings.value('chat_id',''))); self.chat.setReadOnly(True)
        form.addRow('Подключённый личный чат',self.chat)
        self.token.textEdited.connect(lambda:self.chat.clear())
        self.find=QPushButton('Найти мой чат'); layout.addWidget(self.find)
        self.find.clicked.connect(self.connect_chat)
        self.test=QPushButton('Отправить тестовое уведомление'); layout.addWidget(self.test)
        self.test.clicked.connect(self.send_test)
        self.options={}
        for key,label in [('enabled','Включить уведомления'),('finished','Завершение расчёта'),
                          ('early_stop','Ранняя остановка'),('cancelled','Ручная остановка'),('error','Ошибка расчёта')]:
            widget=QCheckBox(label); widget.setChecked(service.settings.value(key,key!='enabled',type=bool))
            self.options[key]=widget; layout.addWidget(widget)
        self.info=QLabel('Windows: токен хранится в диспетчере учётных данных.' if sys.platform=='win32'
                         else 'В этой ОС токен хранится только до закрытия программы.')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        privacy=QLabel('Отправляются статус, время и ошибка модели. Таблицы, координаты и файлы модели не передаются.')
        privacy.setWordWrap(True); layout.addWidget(privacy)
        self.buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setText('Сохранить')
        self.buttons.button(QDialogButtonBox.Cancel).setText('Отмена')
        self.buttons.accepted.connect(self.save); self.buttons.rejected.connect(self.reject); root.addWidget(self.buttons)
        self.forget=QPushButton('Удалить подключение и токен'); layout.addWidget(self.forget)
        self.forget.clicked.connect(self.disconnect)

    def request(self,function,callback):
        for widget in (self.find,self.test,self.token,self.forget,self.buttons): widget.setEnabled(False)
        self.info.setText('Обращение к Telegram…')
        def done(value,error):
            if sip.isdeleted(self): return
            for widget in (self.find,self.test,self.token,self.forget,self.buttons): widget.setEnabled(True)
            if error: self.info.setText(error)
            else: callback(value)
        self.service.submit(function,done)

    def connect_chat(self):
        token=self.token.text().strip()
        def connected(chat):
            self.chat.setText(chat); self.info.setText('Личный чат найден. Отправьте тест и нажмите «Сохранить».')
        self.request(lambda:find_chat(token,self.code),connected)

    def send_test(self):
        token,chat=self.token.text().strip(),self.chat.text()
        if not chat: self.info.setText('Сначала нажмите «Найти мой чат».'); return
        self.request(lambda:api_call(token,'sendMessage',chat_id=chat,text='GeoGarModel: тестовое уведомление. Подключение работает.'),
                     lambda _:self.info.setText('Telegram принял сообщение. Проверьте уведомление на телефоне.'))

    def save(self):
        try:
            if self.options['enabled'].isChecked() and (not self.chat.text() or not self.token.text().strip()):
                raise ValueError('Сначала подключите личный чат.')
            self.service.store.set(self.token.text().strip())
            self.service.settings.setValue('chat_id',self.chat.text())
            for key,widget in self.options.items(): self.service.settings.setValue(key,widget.isChecked())
            self.service.settings.sync(); self.accept()
        except Exception as exc: QMessageBox.warning(self,'Telegram',str(exc))

    def disconnect(self):
        try:
            self.service.store.set(''); self.service.settings.clear(); self.service.settings.sync()
            self.accept()
        except Exception as exc: QMessageBox.warning(self,'Telegram',str(exc))

import json
import os
import sys
import uuid
from types import SimpleNamespace
from urllib.error import URLError
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from PyQt5.QtCore import QSettings,QEventLoop,QTimer
from PyQt5.QtWidgets import QApplication
from geogarmodel import telegram_notifications as tn


@pytest.fixture
def service(tmp_path):
    app=QApplication.instance() or QApplication([])
    class Store:
        value='123:fake_token'
        def get(self): return self.value
        def set(self,value): self.value=value
    settings=QSettings(str(tmp_path/'notifications.ini'),QSettings.IniFormat)
    return app,tn.TelegramNotifications(settings=settings,store=Store())


def test_connection_only_accepts_own_private_code(monkeypatch):
    updates=[{'message':{'chat':{'type':'private','id':11},'text':'/start wrong'}},
             {'message':{'chat':{'type':'group','id':22},'text':'/start secret'}},
             {'message':{'chat':{'type':'private','id':33},'text':'/start secret'}}]
    monkeypatch.setattr(tn,'api_call',lambda *a,**k:updates)
    assert tn.find_chat('123:token','secret')=='33'
    with pytest.raises(ValueError,match='Код не найден'): tn.find_chat('123:token','unknown')


def test_network_errors_never_expose_token(monkeypatch):
    def fail(*a,**k): raise URLError('https://api.telegram.org/bot123:secret_token/sendMessage')
    monkeypatch.setattr(tn,'urlopen',fail)
    with pytest.raises(ValueError) as exc: tn.api_call('123:secret_token','sendMessage',chat_id='33',text='тест')
    assert 'secret_token' not in str(exc.value)
    assert 'Telegram недоступен' in str(exc.value)


def test_post_uses_utf8_and_timeout(monkeypatch):
    import io
    seen=[]
    def open_request(request,timeout):
        seen.append((request,timeout)); return io.BytesIO(b'{"ok":true,"result":{"message_id":1}}')
    monkeypatch.setattr(tn,'urlopen',open_request)
    assert tn.api_call('123:token','sendMessage',chat_id='33',text='Готово')['message_id']==1
    request,timeout=seen[0]
    assert json.loads(request.data)['text']=='Готово' and timeout==8
    assert request.method=='POST'


def test_optional_events_and_background_delivery(service,monkeypatch):
    app,s=service; sent=[]
    monkeypatch.setattr(tn,'api_call',lambda *a,**k:sent.append(k) or {})
    s.send('finished','ignored'); assert not sent
    s.settings.setValue('enabled',True); s.settings.setValue('chat_id','33')
    s.settings.setValue('finished',False)
    s.send('finished','ignored'); assert not sent
    loop=QEventLoop(); results=[]
    s.report=lambda value,error:(results.append(error),loop.quit())
    s.send('early_stop','готово'); QTimer.singleShot(3000,loop.quit); loop.exec_()
    assert results==[''] and sent==[dict(chat_id='33',text='готово')]


@pytest.mark.skipif(sys.platform!='win32',reason='Windows Credential Manager')
def test_windows_credential_roundtrip():
    store=tn.TokenStore('GeoGarModel/Test/'+uuid.uuid4().hex)
    try:
        assert store.get()==''
        store.set('123:test_only'); assert store.get()=='123:test_only'
        store.set('456:replacement'); assert store.get()=='456:replacement'
    finally: store.set('')
    assert store.get()==''


def test_result_message_uses_selected_iteration():
    result=SimpleNamespace(settings={'mode':'gravity','selected_iteration':2,'run_elapsed_seconds':61},
                           history=[{'iteration':2,'normalized_rms':1.2},{'iteration':4,'normalized_rms':1.3}],
                           status='cancelled',mesh=SimpleNamespace(active=np.ones(8,dtype=bool)))
    text=tn.result_message(result)
    assert 'Выбрана модель итерации: 2' in text and 'Завершено итераций: 4' in text
    assert 'Остановлено пользователем' in text and '1.2' in text


def test_dialog_save_never_puts_token_in_settings(service):
    from geogarmodel.telegram_ui import TelegramDialog
    app,s=service; dialog=TelegramDialog(s,None)
    dialog.chat.setText('33'); dialog.options['enabled'].setChecked(True); dialog.save()
    assert s.settings.value('enabled',type=bool)
    assert s.settings.value('chat_id')=='33'
    assert all('token' not in key for key in s.settings.allKeys())
    dialog.disconnect(); assert s.store.value=='' and s.settings.allKeys()==[]


def test_only_live_inversion_notifies(service):
    from geogarmodel.app import MainWindow
    app,_=service; window=MainWindow(); tab=window.inversion
    messages=[]
    window.notifications=SimpleNamespace(send=lambda event,text:messages.append((event,text)))
    tab.demo(); tab.iterations.setValue(2); tab.target.setValue(0)
    loop=QEventLoop(); timer=QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    tab.run(); tab.worker.finished.connect(loop.quit); timer.start(30000); loop.exec_()
    assert tab.worker is None and tab.result is not None
    assert len(messages)==1 and messages[0][0]=='finished'
    # Opening an existing result and editor failures must not send a second notification.
    tab.finished_result(tab.result,offer_export=False); tab.failure('Import error')
    assert len(messages)==1
    for status in ('early_stop','cancelled'):
        tab._notify_inversion=True; tab.result.status=status
        tab.finished_result(tab.result,offer_export=False)
        assert messages[-1][0]==status
    tab._notify_inversion=True; tab.failure('Private filename and data')
    assert messages[-1][0]=='error' and 'Private' not in messages[-1][1]
    window.close(); app.processEvents()

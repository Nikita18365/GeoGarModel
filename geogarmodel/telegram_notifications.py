"""Optional outbound notifications. No survey arrays or model files leave the PC."""
import ctypes
from ctypes import wintypes
import json
import re
import sys
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PyQt5.QtCore import QObject, QSettings, pyqtSignal
from .inversion_progress import duration


class TokenStore:
    """Windows generic credential; other platforms keep the token for this session."""
    def __init__(self,target='GeoGarModel/Telegram'):
        self.session = ''; self.target=target

    def credential_api(self):
        class Credential(ctypes.Structure):
            _fields_ = [('Flags',wintypes.DWORD),('Type',wintypes.DWORD),
                        ('TargetName',wintypes.LPWSTR),('Comment',wintypes.LPWSTR),
                        ('LastWritten',wintypes.FILETIME),('CredentialBlobSize',wintypes.DWORD),
                        ('CredentialBlob',ctypes.POINTER(ctypes.c_ubyte)),('Persist',wintypes.DWORD),
                        ('AttributeCount',wintypes.DWORD),('Attributes',ctypes.c_void_p),
                        ('TargetAlias',wintypes.LPWSTR),('UserName',wintypes.LPWSTR)]
        api = ctypes.WinDLL('Advapi32.dll', use_last_error=True)
        api.CredWriteW.argtypes = [ctypes.POINTER(Credential),wintypes.DWORD]
        api.CredReadW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(ctypes.POINTER(Credential))]
        api.CredDeleteW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD]
        api.CredFree.argtypes = [ctypes.c_void_p]
        return api, Credential

    def get(self):
        if sys.platform != 'win32': return self.session
        api, cls = self.credential_api(); ptr = ctypes.POINTER(cls)()
        if not api.CredReadW(self.target,1,0,ctypes.byref(ptr)):
            if ctypes.get_last_error() == 1168: return ''
            raise ValueError('Не удалось прочитать токен из диспетчера учётных данных Windows.')
        try: return ctypes.string_at(ptr.contents.CredentialBlob,ptr.contents.CredentialBlobSize).decode('utf-8')
        finally: api.CredFree(ptr)

    def set(self, token):
        if sys.platform != 'win32': self.session = token; return
        api, cls = self.credential_api()
        if not token:
            if not api.CredDeleteW(self.target,1,0) and ctypes.get_last_error()!=1168:
                raise ValueError('Не удалось удалить токен из хранилища Windows.')
            return
        raw = token.encode('utf-8'); blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        cred = cls(Type=1,TargetName=self.target,CredentialBlobSize=len(raw),
                   CredentialBlob=blob,Persist=2,UserName='Telegram bot')
        if not api.CredWriteW(ctypes.byref(cred),0):
            raise ValueError('Не удалось сохранить токен в диспетчере учётных данных Windows.')


def api_call(token, method, **parameters):
    if not re.fullmatch(r'[0-9]+:[A-Za-z0-9_-]+',token):
        raise ValueError('Проверьте токен, полученный у BotFather.')
    request = Request(f'https://api.telegram.org/bot{token}/{method}',
                      data=json.dumps(parameters).encode('utf-8'),
                      headers={'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(request, timeout=8) as response: payload=json.load(response)
    except HTTPError as exc:
        messages={401:'Токен Telegram недействителен.',403:'Бот заблокирован: откройте его и нажмите Start.',
                  409:'Бот занят другим подключением или webhook. Используйте отдельного бота.',
                  429:'Telegram ограничил частоту запросов. Попробуйте позднее.'}
        raise ValueError(messages.get(exc.code,'Telegram отклонил запрос. Проверьте токен и чат.')) from None
    except Exception:
        # URL and exception strings can contain the bot token: never surface them.
        raise ValueError('Telegram недоступен: проверьте интернет и повторите попытку.') from None
    if not payload.get('ok'): raise ValueError('Telegram не подтвердил запрос. Проверьте настройки бота.')
    return payload['result']


def find_chat(token, code):
    updates=api_call(token,'getUpdates',offset=-100,limit=100,timeout=0)
    for update in reversed(updates):
        message=update.get('message',{}); chat=message.get('chat',{})
        if chat.get('type')=='private' and message.get('text','').strip()==f'/start {code}':
            return str(chat['id'])
    raise ValueError('Код не найден. Отправьте указанную команду своему боту в личном чате и повторите поиск.')


def result_message(result):
    selected=result.settings.get('selected_iteration',result.history[-1]['iteration'])
    row=next(r for r in result.history if r['iteration']==selected)
    reasons={'target_rms':'Достигнута целевая ошибка','iteration_limit':'Лимит итераций',
             'early_stop':'Ранняя остановка','cancelled':'Остановлено пользователем',
             'stationary':'Малое обновление модели','line_search_limit':'Не найден улучшающий шаг'}
    modes={'gravity':'Плотностное моделирование','induced':'Моделирование магнитной восприимчивости',
           'vector':'Моделирование векторной намагниченности'}
    elapsed=result.settings.get('run_elapsed_seconds',result.settings.get('elapsed_seconds'))
    return (f'GeoGarModel: расчёт завершён\n{modes.get(result.settings["mode"],"3D-моделирование")}\n'
            f'Статус: {reasons.get(result.status,result.status)}\n'
            f'Завершено итераций: {result.history[-1]["iteration"]}\n'
            f'Выбрана модель итерации: {selected}\nRMS/σ: {row["normalized_rms"]:.5g}\n'
            f'Время: {duration(elapsed)}\nАктивных ячеек: {result.mesh.active.sum():,}\n'
            'Результат готов в программе. Сохранение на диск выполняется отдельно.')


class TelegramNotifications(QObject):
    completed=pyqtSignal(object, object, str)
    def __init__(self,parent=None,settings=None,store=None):
        super().__init__(parent)
        self.settings=settings if settings is not None else QSettings('GeoGarModel','Notifications')
        self.store=store if store is not None else TokenStore()
        self.completed.connect(self._completed)

    def submit(self, function, callback):
        def work():
            try: value,error=function(),''
            except Exception as exc: value,error=None,str(exc)
            try: self.completed.emit(callback,value,error)
            except RuntimeError: pass  # Application may have closed during the request.
        Thread(target=work,daemon=True,name='GeoGarModel-Telegram').start()

    def _completed(self,callback,value,error): callback(value,error)

    def enabled(self,event):
        return self.settings.value('enabled',False,type=bool) and self.settings.value(event,True,type=bool)

    def send(self,event,text):
        if not self.enabled(event): return
        try:
            token=self.store.get(); chat=self.settings.value('chat_id','')
            if not token or not chat: raise ValueError('Подключите бота в настройках Telegram.')
        except Exception as exc:
            self.report(None,str(exc)); return
        self.submit(lambda:api_call(token,'sendMessage',chat_id=chat,text=text),self.report)

    def report(self,value,error):
        if self.parent() is not None:
            message='Telegram: '+(error or 'уведомление отправлено')
            self.parent().statusBar().showMessage(message,15000)
            self.parent().inversion.log.appendPlainText(message)

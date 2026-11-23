from io import BytesIO
import json
from urllib.error import URLError

import pytest

from geogarmodel import manual_assistant as helpdesk


def test_docs_find_raster_and_reject_unrelated():
    sources=helpdesk.find_passages('Как привязать растр по точкам?')
    assert any(p.file=='MAP_PROJECTS.md' and 'Импорт изображения' in p.heading for p in sources[:2])
    assert not helpdesk.find_passages('Какая погода в Москве?')
    assert not helpdesk.find_passages('GeoGarModel, как сварить макароны?')
    assert any('FLT' in p.text for p in helpdesk.find_passages('Почему не открывается FLT?'))


@pytest.mark.parametrize(('device','layers'),[('CPU',0),('GPU',-1)])
def test_ollama_only_selects_local_passages(monkeypatch,device,layers):
    sources=[helpdesk.Passage('guide.md','Title','Document text')]

    class Local:
        def open(self,request,timeout):
            assert request.full_url=='http://127.0.0.1:11434/api/chat'
            params=json.loads(request.data)
            assert params['options']['num_gpu']==layers
            assert params['options']['num_ctx']==4096 and params['keep_alive']==0
            return BytesIO(b'{"message":{"content":"{\\"sources\\":[1],\\"outside\\":false}"}}')

    monkeypatch.setattr(helpdesk,'build_opener',lambda *_:Local())
    assert helpdesk.choose_passages('question',sources,device)==sources
    def offline(*args):
        raise URLError('offline')
    monkeypatch.setattr(helpdesk,'build_opener',offline)
    assert helpdesk.choose_passages('question',sources,device) is None


def test_chat_is_accessible_from_header_and_refuses_off_topic(qt_application_lifetime):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QTextOption
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QPushButton
    from geogarmodel.app import MainWindow
    from geogarmodel.manual_assistant_ui import USER_MESSAGE_COLOR

    window=MainWindow()
    try:
        button=next(button for button in window.findChildren(QPushButton) if '🦊' in button.text())
        assert window.manual_assistant.isHidden()
        button.click()
        chat=window.manual_assistant
        assert not chat.isHidden()
        assert [chat.device.itemText(i) for i in range(chat.device.count())]==['CPU','GPU']
        assert chat.input.wordWrapMode()==QTextOption.WrapAtWordBoundaryOrAnywhere
        assert chat.history.wordWrapMode()==QTextOption.WrapAtWordBoundaryOrAnywhere
        chat.input.setPlainText('Какая погода в Москве?')
        chat.input.moveCursor(chat.input.textCursor().End)
        QTest.keyClick(chat.input,Qt.Key_Return,Qt.ShiftModifier)
        chat.input.insertPlainText('<'+('a'*160)+'>')
        QTest.keyClick(chat.input,Qt.Key_Return)
        assert not chat.input.toPlainText()
        assert helpdesk.OFF_TOPIC in chat.history.toPlainText()
        assert USER_MESSAGE_COLOR in chat.history.toHtml()
        assert '<'+('a'*160)+'>' in chat.history.toPlainText()
    finally:
        window.close()

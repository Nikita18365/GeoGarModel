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
            return BytesIO(json.dumps({'message':{'content':json.dumps(
                {'text':'Ответ по руководству','sources':[1],'outside':False})}}).encode())

    monkeypatch.setattr(helpdesk,'build_opener',lambda *_:Local())
    assert helpdesk.choose_passages('question',sources,device)==helpdesk.Answer('Ответ по руководству',tuple(sources),False)
    def offline(*args):
        raise URLError('offline')
    monkeypatch.setattr(helpdesk,'build_opener',offline)
    assert helpdesk.choose_passages('question',sources,device) is None


def test_sigma_looks_up_explanation_and_supplies_web_source(monkeypatch):
    question='Что такое «сигма константа» в 3Д моделировании «1. Поле»? Сейчас она равна 1.00.'
    sources=helpdesk.find_passages(question)
    assert 'Что означает «σ константа»' in sources[0].heading
    assert '1 мГал' in sources[0].text and '1 нТл' in sources[0].text
    web=helpdesk.Passage('Википедия','Взвешенный МНК','Вес данных','https://ru.wikipedia.org/?curid=1')
    queries=[]
    monkeypatch.setattr(helpdesk,'search_web',lambda topic:queries.append(topic) or (web,))

    class Local:
        def open(self,request,timeout):
            params=json.loads(request.data)
            context=json.loads(params['messages'][-1]['content'])
            assert context['question']==question
            assert context['excerpts'][-1]['kind']=='Интернет'
            ids=[1,context['excerpts'][-1]['id']]
            return BytesIO(json.dumps({'message':{'content':json.dumps(
                {'text':'Ожидаемая погрешность наблюдений','sources':ids,'outside':False})}}).encode())

    monkeypatch.setattr(helpdesk,'build_opener',lambda *_:Local())
    answer=helpdesk.choose_passages(question,sources,'CPU',use_web=True)
    assert answer.sources==(sources[0],web)
    assert queries==['Метод наименьших квадратов взвешенный']


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


def test_variable_height_question_gets_direct_answer_in_chat(qt_application_lifetime,monkeypatch):
    from geogarmodel.app import MainWindow
    from geogarmodel import manual_assistant_ui

    def unexpected_request(*args,**kwargs):
        pytest.fail('Вопрос о переменной высоте должен обрабатываться без вызова Ollama.')
    monkeypatch.setattr(manual_assistant_ui,'ManualRequest',unexpected_request)
    question='Возможно ли 3D-моделирование плотности в данной программе, если наблюдения выполнялись на переменной высоте, не на константе'
    window=MainWindow()
    try:
        chat=window.manual_assistant
        chat.input.setPlainText(question)
        chat.ask()
        transcript=chat.history.toPlainText()
        assert transcript.count(question)==1
        assert 'Да. Для наблюдений на разных высотах' in transcript
        assert 'Прямой расчёт по призмам' in transcript
        assert 'Высота Up, м' in transcript
        assert 'требует, чтобы все наблюдения лежали на одной высоте' in transcript
        assert chat.sources[0].file=='INVERSION.md'
        fft=helpdesk.verified_answer('Можно ли выполнить 2D FFT, если высоты разные?')
        assert fft.text.startswith('**Для послойной 2D FFT — нет.**')
        assert helpdesk.strip_question_echo('**'+question+'**\n\nДа, методом призм.',question)=='Да, методом призм.'
        assert helpdesk.strip_question_echo(question,question)==''
    finally:
        window.close()


def test_assistant_markdown_preserves_styles_and_numbered_steps(qt_application_lifetime):
    from PyQt5.QtGui import QFont,QTextCursor
    from geogarmodel.app import MainWindow

    window=MainWindow()
    try:
        chat=window.manual_assistant
        chat.append_markdown('5. Введите **Целевую RMS / σ**.\n'
                             '6. Нажмите *Запустить 3D-моделирование*.\n\n'
                             '[Поддельный источник](help:999) ![Фото](file:///tmp/photo.png)')
        doc=chat.history.document()
        assert doc.defaultFont().family() in ('Inter','OpenAI Sans')
        assert chat.input.font().family()==doc.defaultFont().family()
        bold=doc.find('Целевую RMS / σ')
        italic=doc.find('Запустить 3D-моделирование')
        assert bold.charFormat().fontWeight()>=QFont.Bold
        assert italic.charFormat().fontItalic()
        assert '5. Введите' in doc.toPlainText()
        assert '6. Нажмите' in doc.toPlainText()
        assert not doc.find('Поддельный источник').charFormat().isAnchor()
        assert '\ufffc' not in doc.toPlainText()
    finally:
        window.close()


def test_pet_click_drag_and_visibility(qt_application_lifetime,tmp_path):
    from PyQt5.QtCore import QPoint,QPointF,QSettings,Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication,QWidget
    from geogarmodel.assistant_pet import AssistantPet,PETS

    host=QWidget()
    host.resize(600,500)
    settings=QSettings(str(tmp_path/'pet.ini'),QSettings.IniFormat)
    clicks=[]
    greetings=[]
    pet=AssistantPet(host,settings,lambda:clicks.append(True))
    try:
        host.show()
        QApplication.processEvents()
        assert pet.isVisible() and pet.timer.isActive()
        center=QPoint(46,49)
        start=pet.mapToGlobal(center)
        for movement in (QPoint(),QPoint(-45,-25)):
            origin=pet.pos()
            pet.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress,QPointF(center),QPointF(start),
                                           Qt.LeftButton,Qt.LeftButton,Qt.NoModifier))
            pet.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease,QPointF(center+movement),
                                             QPointF(start+movement),Qt.LeftButton,Qt.NoButton,Qt.NoModifier))
        assert clicks==[True]
        assert pet.greeting.text()=='Лиса · '+PETS['fox'][1]
        assert pet.greeting.isVisible()
        assert pet.greeting_timer.interval()==7000
        assert pet.greeting_timer.isActive()
        assert pet.pos()==pet.on_screen(origin+movement)
        assert settings.value('assistant/petPosition')==pet.pos()
        assert len(pet.characters.actions())==6
        for choice,(key,(_,greeting)) in zip(pet.characters.actions(),PETS.items()):
            choice.trigger()
            assert not pet.grab().isNull()
            assert settings.value('assistant/pet')==key
            point=pet.mapToGlobal(center)
            pet.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress,QPointF(center),QPointF(point),
                                           Qt.LeftButton,Qt.LeftButton,Qt.NoModifier))
            pet.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease,QPointF(center),QPointF(point),
                                             Qt.LeftButton,Qt.NoButton,Qt.NoModifier))
            greetings.append(pet.greeting.text())
            assert greetings[-1]==f'{choice.text()} · {greeting}'
            assert pet.reaction>0
        assert len(set(greetings))==6
        from PyQt5.QtTest import QTest
        QApplication.sendEvent(pet,QEvent(QEvent.Leave))
        QTest.qWait(6500)
        assert pet.greeting.isVisible()
        QTest.qWait(700)
        assert not pet.greeting.isVisible()
        pet.dismiss.click()
        assert pet.isHidden() and not pet.timer.isActive()
        pet.visibility_action.trigger()
        assert pet.isVisible()
    finally:
        host.close()

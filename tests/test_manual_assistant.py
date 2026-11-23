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


@pytest.mark.parametrize(('device','layers','model'),[('CPU',0,'qwen3:4b'),('GPU',-1,'qwen3.8:27b')])
def test_ollama_only_selects_local_passages(monkeypatch,device,layers,model):
    sources=[helpdesk.Passage('guide.md','Title','Document text')]

    class Local:
        def open(self,request,timeout):
            assert request.full_url=='http://127.0.0.1:11434/api/chat'
            params=json.loads(request.data)
            assert params['model']==model and params['think'] is False
            assert timeout==120
            assert params['options']['num_gpu']==layers
            assert params['options']['num_ctx']==4096 and params['keep_alive']==0
            return BytesIO(json.dumps({'message':{'content':json.dumps(
                {'text':'Ответ по руководству','sources':[1],'outside':False})}}).encode())

    monkeypatch.setattr(helpdesk,'build_opener',lambda *_:Local())
    assert helpdesk.choose_passages('question',sources,device,model=model)==helpdesk.Answer('Ответ по руководству',tuple(sources),False)
    def offline(*args):
        raise URLError('offline')
    monkeypatch.setattr(helpdesk,'build_opener',offline)
    assert helpdesk.choose_passages('question',sources,device) is None


def test_sigma_looks_up_explanation_and_supplies_web_source(monkeypatch):
    question='Что такое «сигма константа» в 3Д моделировании «1. Поле»? Сейчас она равна 1.00.'
    sources=helpdesk.find_passages(question)
    assert 'Что означает «σ константа»' in sources[0].heading
    assert '1 мГал' in sources[0].text and '1 нТл' in sources[0].text
    follow_up = helpdesk.find_passages('А в каких единицах она задаётся?', question)
    assert follow_up[0].heading == sources[0].heading
    web=helpdesk.Passage('Википедия','Взвешенный МНК','Вес данных','https://ru.wikipedia.org/?curid=1')
    queries=[]
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
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
    answer=helpdesk.answer_question(question,'CPU',helpdesk.HYBRID,helpdesk.DEFAULT_MODEL)
    assert answer.sources==(sources[0],web)
    assert queries==['Метод наименьших квадратов взвешенный']


def test_cloud_model_uses_selected_tag_without_schema(monkeypatch):
    source=helpdesk.Passage('guide.md','Title','Text')
    model='kimi-k3.1-pro:cloud'

    class Local:
        def open(self,request,timeout):
            params=json.loads(request.data)
            assert params['model']==model
            assert 'format' not in params and 'options' not in params
            return BytesIO(json.dumps({'message':{'content':
                '```json\n{"text":"Ответ","sources":[1],"outside":false}\n```'}}).encode())

    monkeypatch.setattr(helpdesk,'build_opener',lambda *_:Local())
    answer=helpdesk.choose_passages('Вопрос',[source],'CPU',model=model)
    assert answer==helpdesk.Answer('Ответ',(source,),False)


def test_source_modes_and_web_fallback(monkeypatch):
    question = 'Зачем задавать сигму при моделировании плотности?'
    web = helpdesk.Passage('example.org', 'Weights', 'Measurement weights', 'https://example.org/weights')
    searches = []
    picked = []
    monkeypatch.setenv('OLLAMA_API_KEY', 'test-key')
    monkeypatch.setattr(helpdesk, 'search_ollama', lambda query, key: searches.append((query, key)) or (web,))

    def reply(question, sources, device, previous, **kwargs):
        picked.append(sources)
        return helpdesk.Answer('Ответ с источниками', tuple(sources), False)

    monkeypatch.setattr(helpdesk, 'choose_passages', reply)
    for mode in (helpdesk.LOCAL, helpdesk.INTERNET, helpdesk.HYBRID):
        helpdesk.answer_question(question, 'CPU', mode, helpdesk.DEFAULT_MODEL)
    assert all(not passage.url for passage in picked[0])
    assert picked[1] == [web]
    assert picked[2][-1] == web and any(not passage.url for passage in picked[2])
    assert searches == [(question, 'test-key')] * 2
    monkeypatch.setattr(helpdesk, 'choose_passages', lambda question, sources, *args, **kwargs:
                        helpdesk.Answer('Ответ по руководству', (sources[0],), False))
    mixed = helpdesk.answer_question(question, 'CPU', helpdesk.HYBRID, helpdesk.DEFAULT_MODEL)
    assert mixed.supplement == (web,)

    def offline(*args):
        raise URLError('offline')

    monkeypatch.setattr(helpdesk, 'search_ollama', offline)
    monkeypatch.setattr(helpdesk, 'search_web', offline)
    answer = helpdesk.answer_question(question, 'CPU', helpdesk.INTERNET, helpdesk.DEFAULT_MODEL)
    assert not answer.sources and 'Внешние источники не получены' in answer.notice
    answer = helpdesk.answer_question(question, 'CPU', helpdesk.HYBRID, helpdesk.DEFAULT_MODEL)
    assert answer.sources and all(not passage.url for passage in answer.sources)


def test_long_chat_recalls_relevant_turns_and_excludes_fallbacks():
    from geogarmodel.assistant_history import Message, conversation_context

    source = helpdesk.Passage('INVERSION.md', 'Сигма', 'Погрешность наблюдений')
    messages = [
        Message('user', 'Сигма в плотностном моделировании', helpdesk.LOCAL),
        Message('assistant', 'Сигма — погрешность', helpdesk.LOCAL, (source,)),
        *(Message('user', f'Как оформить карту {i}?', helpdesk.LOCAL) for i in range(40)),
        Message('assistant', 'Не удалось ответить', helpdesk.LOCAL, (source,), excerpt=True),
    ]
    context = conversation_context(messages, 'Как задать сигму?', helpdesk.LOCAL)
    assert len(context) == 3
    assert context[0]['answer'] == 'Сигма — погрешность'
    assert 'answer' not in context[-1]
    context = conversation_context(messages, 'Как задать сигму?', helpdesk.INTERNET)
    assert 'answer' not in context[0]


def test_chat_history_reopens_and_keeps_replies_in_origin_chat(qt_application_lifetime, monkeypatch):
    from PyQt5.QtCore import QThreadPool
    from PyQt5.QtWidgets import QInputDialog, QMessageBox
    from geogarmodel.app import MainWindow
    from geogarmodel import manual_assistant_ui

    contexts = []
    source = helpdesk.Passage('INVERSION.md', 'Сигма', 'Сигма — погрешность наблюдений.')

    def respond(question, device, mode, model, context):
        contexts.append(context)
        return helpdesk.Answer('Погрешность наблюдений.', (source,), False)

    monkeypatch.setattr(manual_assistant_ui, 'answer_question', respond)
    window = MainWindow()
    try:
        chat = window.manual_assistant
        chat.mode.setCurrentIndex(chat.mode.findData(helpdesk.LOCAL))
        first = chat.chats.currentData()
        chat.input.setPlainText('Что такое сигма?')
        chat.ask()
        chat.new_chat()
        second = chat.chats.currentData()
        assert first != second
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
        assert 'Погрешность наблюдений.' not in chat.history.toPlainText()
        chat.chats.setCurrentIndex(chat.chats.findData(first))
        monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Параметры инверсии', True))
        chat.rename_chat()
    finally:
        window.close()

    window = MainWindow()
    try:
        chat = window.manual_assistant
        assert chat.chats.currentText() == 'Параметры инверсии'
        assert chat.mode.currentData() == helpdesk.LOCAL
        assert 'Погрешность наблюдений.' in chat.history.toPlainText()
        assert chat.sources == [source]
        chat.input.setPlainText('А в каких единицах она задаётся?')
        chat.ask()
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
        assert contexts == [[], [{'question': 'Что такое сигма?', 'answer': 'Погрешность наблюдений.'}]]
        monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
        chat.delete_chat()
        assert chat.chats.currentData() == second and chat.store.messages(first) == []
    finally:
        window.close()


def test_model_picker_displays_model_specific_commands(qt_application_lifetime,monkeypatch):
    from geogarmodel.app import MainWindow
    from geogarmodel.ollama_models_ui import OllamaModels
    import geogarmodel.ollama_models_ui as model_ui
    window=MainWindow()
    try:
        dialog=OllamaModels(window,'qwen3:4b')
        monkeypatch.setattr(model_ui,'installed_models',lambda:['local-model:latest'])
        dialog.refresh_installed()
        assert dialog.models.findText('local-model:latest')>=0
        dialog.models.setCurrentText('namespace/example:custom')
        assert dialog.ok.isEnabled()
        assert dialog.commands.toPlainText()=='ollama pull namespace/example:custom'
        dialog.models.setCurrentText('deepseek-v4.1-flash:cloud')
        assert dialog.commands.toPlainText()=='ollama signin\nollama run deepseek-v4.1-flash:cloud'
        assert 'Ollama Cloud' in dialog.note.text()
        dialog.models.setCurrentText('ollama pull invalid')
        assert not dialog.ok.isEnabled()
    finally:
        window.close()


def test_chat_is_accessible_from_header_and_refuses_off_topic(qt_application_lifetime):
    from PyQt5.QtCore import Qt, QThreadPool
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
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
        assert not chat.input.toPlainText()
        assert helpdesk.OFF_TOPIC in chat.history.toPlainText()
        assert USER_MESSAGE_COLOR in chat.history.toHtml()
        assert '<'+('a'*160)+'>' in chat.history.toPlainText()
    finally:
        window.close()


def test_variable_height_question_gets_direct_answer_in_chat(qt_application_lifetime,monkeypatch):
    from PyQt5.QtCore import QThreadPool
    from geogarmodel.app import MainWindow

    def unexpected_request(*args,**kwargs):
        pytest.fail('Вопрос о переменной высоте должен обрабатываться без вызова Ollama.')
    monkeypatch.setattr(helpdesk,'choose_passages',unexpected_request)
    question='Возможно ли 3D-моделирование плотности в данной программе, если наблюдения выполнялись на переменной высоте, не на константе'
    window=MainWindow()
    try:
        chat=window.manual_assistant
        chat.mode.setCurrentIndex(chat.mode.findData(helpdesk.LOCAL))
        chat.input.setPlainText(question)
        chat.ask()
        assert QThreadPool.globalInstance().waitForDone(3000)
        qt_application_lifetime.processEvents()
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

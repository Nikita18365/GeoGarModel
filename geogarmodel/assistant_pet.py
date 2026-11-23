"""A floating GeoGarModel mascot that opens the manual chat."""
from math import sin

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QActionGroup, QApplication, QMenu, QToolButton, QToolTip, QWidget


PETS = {
    'fox': ('Лиса', 'Шур-шур! Какой след найдём в руководстве?'),
    'owl': ('Сова', 'Угу! Что в GeoGarModel разберём по шагам?'),
    'robot': ('Робот', 'Бип-бип! Готов искать нужную команду.'),
    'cat': ('Кот', 'Мяу, что сегодня изучим?'),
    'paperclip': ('Скрепка', 'Щёлк! Какой раздел руководства открыть?'),
    'seal': ('Тюлень', 'Плюх! В какую тему нырнём?'),
}


class AssistantPet(QWidget):
    def __init__(self, window, settings, toggle_chat):
        super().__init__(window,Qt.Tool|Qt.FramelessWindowHint|Qt.WindowDoesNotAcceptFocus)
        self.host = window
        self.settings = settings
        self.toggle_chat = toggle_chat
        self.character = settings.value('assistant/pet','fox',type=str)
        if self.character not in PETS:
            self.character = 'fox'
        self.enabled = settings.value('assistant/petVisible',True,type=bool)
        self.press_position = None
        self.dragging = False
        self.frame = 0
        self.reaction = 0
        self.setObjectName('assistantPet')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setStyleSheet('QWidget#assistantPet {background: transparent;}')
        self.setFixedSize(92,94)
        self.setCursor(Qt.OpenHandCursor)
        self.setAccessibleName('Питомец GeoGarModel')
        self.setToolTip('Нажмите, чтобы открыть справку. Перетащите, чтобы переместить.')
        self.dismiss = QToolButton(self)
        self.dismiss.setText('×')
        self.dismiss.setAccessibleName('Скрыть питомца')
        self.dismiss.setFixedSize(21,21)
        self.dismiss.move(70,1)
        self.dismiss.setStyleSheet('QToolButton {padding: 0; border: 1px solid #34506b; '
                                   'background: #182b43; border-radius: 10px;}')
        self.dismiss.clicked.connect(lambda: self.set_pet_visible(False))
        self.timer = QTimer(self)
        self.timer.setInterval(75)
        self.timer.timeout.connect(self.animate)
        self.menu = QMenu(window)
        self.characters = QActionGroup(self.menu)
        self.characters.setExclusive(True)
        for key,(label,_) in PETS.items():
            choice = self.menu.addAction(label)
            choice.setCheckable(True)
            choice.setChecked(key==self.character)
            choice.triggered.connect(lambda _,name=key:self.choose(name))
            self.characters.addAction(choice)
        self.menu.addSeparator()
        self.visibility_action = self.menu.addAction('')
        self.visibility_action.triggered.connect(lambda: self.set_pet_visible(not self.enabled))
        self.menu.aboutToShow.connect(self.refresh_menu)
        self.host.installEventFilter(self)
        self.positioned = False

    def refresh_menu(self):
        self.visibility_action.setText('Скрыть питомца' if self.enabled else 'Показать питомца')

    def choose(self,character):
        self.character = character
        self.settings.setValue('assistant/pet',character)
        self.reaction = 0
        self.update()

    def set_pet_visible(self,visible):
        self.enabled = visible
        self.settings.setValue('assistant/petVisible',visible)
        self.sync_visibility()

    def eventFilter(self,watched,event):
        if watched is self.host:
            if event.type() in (QEvent.Show,QEvent.WindowStateChange):
                QTimer.singleShot(0,self.sync_visibility)
            elif event.type()==QEvent.Hide:
                self.hide()
        return super().eventFilter(watched,event)

    def sync_visibility(self):
        if not self.enabled or not self.host.isVisible() or self.host.isMinimized():
            self.hide()
            return
        if not self.positioned:
            point = self.settings.value('assistant/petPosition',None)
            if not isinstance(point,QPoint):
                corner = self.host.mapToGlobal(self.host.rect().bottomRight())
                point = corner-QPoint(self.width()+22,self.height()+40)
            self.move(self.on_screen(point))
            self.positioned = True
        self.show()

    def on_screen(self,point):
        screen = QApplication.screenAt(point+self.rect().center()) or self.host.screen()
        area = screen.availableGeometry()
        return QPoint(max(area.left(),min(point.x(),area.right()-self.width()+1)),
                      max(area.top(),min(point.y(),area.bottom()-self.height()+1)))

    def showEvent(self,event):
        super().showEvent(event)
        self.timer.start()

    def hideEvent(self,event):
        self.timer.stop()
        self.reaction = 0
        self.press_position = None
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor)
        super().hideEvent(event)

    def animate(self):
        self.frame = (self.frame+1)%80
        self.reaction = max(0,self.reaction-1)
        self.update()

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.press_position = event.globalPos()
            self.initial_position = self.pos()
            self.dragging = False
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.press_position is None:
            return
        delta = event.globalPos()-self.press_position
        if self.dragging or delta.manhattanLength()>=QApplication.startDragDistance():
            self.dragging = True
            self.move(self.initial_position+delta)

    def mouseReleaseEvent(self,event):
        if event.button()!=Qt.LeftButton or self.press_position is None:
            return
        delta = event.globalPos()-self.press_position
        self.press_position = None
        self.setCursor(Qt.OpenHandCursor)
        if self.dragging or delta.manhattanLength()>=QApplication.startDragDistance():
            self.move(self.on_screen(self.initial_position+delta))
            self.settings.setValue('assistant/petPosition',self.pos())
        else:
            self.toggle_chat()
            self.reaction = 10
            label,line = PETS[self.character]
            self.setAccessibleDescription(line)
            QToolTip.showText(self.mapToGlobal(QPoint(12,-8)),f'{label} · {line}',self,QRect(),3500)
        self.dragging = False

    def contextMenuEvent(self,event):
        self.menu.popup(event.globalPos())

    def paintEvent(self,event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0,0,0,110))
        painter.drawEllipse(17,79,58,10)
        painter.translate(0,2.4*sin(self.frame*0.157))
        if self.reaction:
            painter.translate(0,-5*sin(self.reaction*.65))
        artist = PetDrawing(painter)
        drawings = {'fox':artist.fox,'owl':artist.owl,'robot':artist.robot,'cat':artist.cat,
                    'paperclip':artist.paperclip,'seal':artist.seal}
        drawings[self.character]()
        artist.eyes(self.frame in (36,37,38) or self.reaction==9)


class PetDrawing:
    def __init__(self,painter):
        self.painter = painter

    def oval(self,color,x,y,w,h):
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(QColor(color))
        self.painter.drawEllipse(x,y,w,h)

    def poly(self,color,*points):
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(QColor(color))
        self.painter.drawPolygon(QPolygonF([QPointF(*point) for point in points]))

    def eyes(self,blink):
        for x in (33,55):
            if blink:
                self.painter.setPen(QPen(QColor('#112138'),2))
                self.painter.drawLine(x-3,51,x+3,51)
            else:
                self.oval('#112138',x-3,46,6,10)
                self.oval('#ffffff',x-1,47,2,2)

    def fox(self):
        self.oval('#bc5428',19,53,54,30)
        self.poly('#dc763d',(16,44),(17,16),(40,34),(53,34),(74,16),(76,45))
        self.poly('#ffd1a1',(22,34),(22,24),(33,36))
        self.poly('#ffd1a1',(59,36),(70,24),(70,34))
        self.oval('#ef914c',13,31,66,42)
        self.poly('#fff0d4',(14,52),(33,58),(46,69),(59,58),(78,52),(70,67),(46,79),(22,67))
        self.oval('#112138',42,59,8,6)

    def owl(self):
        self.poly('#5278a1',(18,40),(18,21),(32,32),(60,32),(74,21),(74,40))
        self.oval('#638ebe',15,29,62,52)
        self.oval('#dcf2fa',21,37,26,27)
        self.oval('#dcf2fa',44,37,26,27)
        self.poly('#e9ad52',(42,58),(50,58),(46,65))
        self.oval('#e9ad52',30,77,12,5)
        self.oval('#e9ad52',50,77,12,5)

    def robot(self):
        self.painter.setPen(QPen(QColor('#6fd3b8'),3))
        self.painter.drawLine(46,19,46,32)
        self.oval('#f1bd68',42,15,8,8)
        self.oval('#468591',11,44,15,22)
        self.oval('#468591',66,44,15,22)
        self.painter.setBrush(QColor('#74c6c6'))
        self.painter.drawRoundedRect(18,31,56,46,14,14)
        self.oval('#dcf7ed',24,39,44,23)
        self.oval('#325464',34,65,24,4)

    def cat(self):
        self.oval('#a36c9e',22,57,54,26)
        self.poly('#a36c9e',(17,51),(18,16),(41,35),(54,35),(73,16),(76,53))
        self.poly('#efb9d0',(23,34),(23,24),(34,36))
        self.poly('#efb9d0',(59,36),(69,24),(69,34))
        self.oval('#c68eba',16,31,60,44)
        self.oval('#f6e1ed',32,54,28,16)
        self.poly('#933f6f',(41,57),(51,57),(46,62))
        self.painter.setPen(QPen(QColor('#693154'),1.5))
        for start,end in (((29,55),(15,51)),((28,59),(14,60)),((63,55),(77,51)),((64,59),(78,60))):
            self.painter.drawLine(QPointF(*start),QPointF(*end))

    def paperclip(self):
        path = QPainterPath(QPointF(59,26))
        path.cubicTo(59,16,49,13,41,17)
        path.cubicTo(34,20,34,27,34,34)
        path.lineTo(34,68)
        path.cubicTo(34,88,65,87,65,68)
        path.lineTo(65,41)
        path.cubicTo(65,29,47,29,47,41)
        path.lineTo(47,67)
        self.painter.setPen(QPen(QColor('#97bce8'),5,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
        self.painter.drawPath(path)
        self.oval('#d6edff',20,41,47,21)
        self.oval('#3383bc',42,64,9,4)

    def seal(self):
        self.poly('#579497',(29,67),(4,77),(21,83),(43,77),(66,77),(88,74),(75,66))
        self.oval('#6bafad',13,32,68,48)
        self.oval('#d9f0e7',31,53,31,19)
        self.oval('#224a56',42,55,9,6)
        self.oval('#f7b8ad',22,55,9,5)
        self.oval('#f7b8ad',62,55,9,5)

"""Mouse-controlled illumination hemisphere: edge is horizon, centre is zenith."""
import math
from PyQt5.QtCore import Qt,QSize,pyqtSignal,QPointF,QTimer
from PyQt5.QtGui import QPainter,QPen,QColor
from PyQt5.QtWidgets import QWidget


class SunPicker(QWidget):
    changed=pyqtSignal(float,float)

    def __init__(self,parent=None):
        super().__init__(parent); self.azimuth=315.; self.altitude=45.
        self.setMinimumHeight(170)
        self.setToolTip('Перетащите солнце: направление задаёт азимут, расстояние от центра — высоту. Центр: 90°, край: 0°.')
        self.timer=QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(80)
        self.timer.timeout.connect(self.publish)

    def sizeHint(self): return QSize(220,180)

    def geometry_values(self):
        return QPointF(self.width()/2,self.height()/2),min(self.width(),self.height())/2-22

    def set_light(self,azimuth,altitude):
        self.azimuth=azimuth; self.altitude=altitude; self.update()

    def publish(self): self.changed.emit(self.azimuth,self.altitude)

    def point(self,event):
        center,radius=self.geometry_values(); dx=event.pos().x()-center.x(); dy=event.pos().y()-center.y()
        length=math.hypot(dx,dy)
        if length>1: self.azimuth=math.degrees(math.atan2(dx,-dy))%360
        self.altitude=90*(1-min(1,length/radius)); self.update()

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.point(event); self.publish()

    def mouseMoveEvent(self,event):
        if event.buttons()&Qt.LeftButton:
            self.point(event)
            if not self.timer.isActive(): self.timer.start()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.timer.stop(); self.point(event); self.publish()

    def paintEvent(self,event):
        painter=QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        center,radius=self.geometry_values(); painter.setPen(QPen(QColor('#526b89'),1)); painter.setBrush(QColor('#14263b'))
        painter.drawEllipse(center,radius,radius)
        painter.setBrush(Qt.NoBrush); painter.setPen(QPen(QColor('#304965'),1,Qt.DashLine))
        painter.drawEllipse(center,radius/2,radius/2)
        painter.drawLine(QPointF(center.x()-radius,center.y()),QPointF(center.x()+radius,center.y()))
        painter.drawLine(QPointF(center.x(),center.y()-radius),QPointF(center.x(),center.y()+radius))
        painter.setPen(QColor('#dbe7f6'))
        for label,dx,dy in [('С',-5,-radius-7),('Ю',-5,radius+16),('З',-radius-16,4),('В',radius+6,4)]:
            painter.drawText(QPointF(center.x()+dx,center.y()+dy),label)
        angle=math.radians(self.azimuth); distance=radius*(1-self.altitude/90)
        sun=QPointF(center.x()+math.sin(angle)*distance,center.y()-math.cos(angle)*distance)
        painter.setPen(QPen(QColor('#fbbf24'),2)); painter.drawLine(center,sun)
        painter.setBrush(QColor('#fde68a')); painter.drawEllipse(sun,7,7)

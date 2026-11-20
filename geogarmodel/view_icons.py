"""Small native vector icons for model navigation, independent of external APIs."""
from PyQt5.QtCore import Qt,QPointF,QRectF
from PyQt5.QtGui import QIcon,QPixmap,QPainter,QPen,QColor,QPolygonF


def view_icon(kind):
    pix=QPixmap(28,28); pix.fill(Qt.transparent); p=QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#c7e3f5'),1.4)); p.setBrush(QColor('#243e58'))
    if kind in ('palette','maps'):
        for i,color in enumerate(['#142dff','#00cee6','#14df38','#fff51b','#ff8314','#ff144c']):
            p.fillRect(QRectF(3+i*3.7,6,4,16),QColor(color))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(3,6,22,16))
        if kind=='maps': p.drawLine(5,10,23,10); p.drawLine(5,15,23,15)
    elif kind=='fit':
        for x,y,sx,sy in [(3,3,1,1),(25,3,-1,1),(3,25,1,-1),(25,25,-1,-1)]:
            p.drawLine(x,y,x+sx*7,y); p.drawLine(x,y,x,y+sy*7)
        p.setBrush(QColor('#37d5c4')); p.drawRect(10,10,8,8)
    elif kind in ('new','load','tile','cascade','remove'):
        if kind=='remove': p.setPen(QPen(QColor('#ff9d9d'),2)); p.drawLine(8,8,21,21); p.drawLine(8,21,21,8)
        elif kind in ('tile','cascade'):
            rects=[(3,4,10,20),(15,4,10,20)] if kind=='tile' else [(3,4,16,15),(9,10,16,15)]
            for r in rects: p.drawRect(*r)
        else:
            p.drawRect(4,6,20,17); p.setPen(QPen(QColor('#37d5c4'),2)); p.drawLine(14,10,14,20)
            if kind=='new': p.drawLine(9,15,19,15)
            else: p.drawLine(10,16,14,20); p.drawLine(14,20,18,16)
    else:
        faces=[[(4,9),(14,3),(24,9),(14,15)],[(4,9),(14,15),(14,26),(4,20)],[(14,15),(24,9),(24,20),(14,26)]]
        face={'top':0,'bottom':0,'front':1,'back':1,'right':2,'left':2,'slice_x':1,'slice_y':2,'slice_z':0}.get(kind)
        for i,points in enumerate(faces):
            p.setBrush(QColor('#37d5c4' if i==face else '#243e58')); p.drawPolygon(QPolygonF([QPointF(*v) for v in points]))
        if kind.startswith('slice_'):
            p.setPen(QPen(QColor('#ffcc67'),2)); p.drawLine(5,14,23,14)
        if kind in ('bottom','back','left'):
            p.setPen(QPen(QColor('#ffcc67'),1.5)); p.drawLine(18,22,25,22); p.drawLine(25,22,22,19)
    p.end(); return QIcon(pix)

"""Small native vector icons for model navigation, independent of external APIs."""
from PyQt5.QtCore import Qt,QPointF,QRectF
from PyQt5.QtGui import QIcon,QPixmap,QPainter,QPen,QColor,QPolygonF,QPainterPath


def view_icon(kind):
    pix=QPixmap(28,28); pix.fill(Qt.transparent); p=QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#c7e3f5'),1.4)); p.setBrush(QColor('#243e58'))
    if kind in ('palette','maps'):
        for i,color in enumerate(['#142dff','#00cee6','#14df38','#fff51b','#ff8314','#ff144c']):
            p.fillRect(QRectF(3+i*3.7,6,4,16),QColor(color))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(3,6,22,16))
        if kind=='maps': p.drawLine(5,10,23,10); p.drawLine(5,15,23,15)
    elif kind=='layer_grid':
        # Colour bands and curved isolines: a numerical geophysical grid.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#213b66')); p.drawRoundedRect(QRectF(2,3,24,22),3,3)
        for i,color in enumerate(('#3678d4','#35a9c7','#51c997','#f2d15b','#e77a45')):
            p.fillRect(QRectF(4+i*4,5,4,18),QColor(color))
        p.setPen(QPen(QColor('#24364b'),1.5)); p.setBrush(Qt.NoBrush)
        for offset in (0,7):
            contour=QPainterPath(QPointF(4,10+offset))
            contour.cubicTo(11,2+offset,15,21+offset,24,10+offset)
            p.drawPath(contour)
        p.setPen(QPen(QColor('#b9d1e5'),1.2))
        p.drawRoundedRect(QRectF(2,3,24,22),3,3)
    elif kind=='layer_vector':
        # Polygon, polyline and a point; one icon for all vector geometries.
        p.setPen(QPen(QColor('#b1ebba'),1.7)); p.setBrush(QColor('#2c714b'))
        p.drawPolygon(QPolygonF([QPointF(4,5),QPointF(15,4),QPointF(20,12),QPointF(7,14)]))
        p.setPen(QPen(QColor('#59dfae'),2.2)); p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF([QPointF(3,22),QPointF(10,17),QPointF(17,23),QPointF(25,15)]))
        p.setPen(QPen(QColor('#efffdc'),1.2)); p.setBrush(QColor('#f2cd69'))
        p.drawEllipse(QRectF(20,3,5,5))
    elif kind=='layer_raster':
        # A framed pixel mosaic, distinct from a scalar grid with isolines.
        p.setPen(Qt.NoPen); p.setBrush(QColor('#364b67'))
        p.drawRoundedRect(QRectF(2,3,24,22),3,3)
        colors=(('#90bce8','#90bce8','#c9d9ee'),
                ('#7893a6','#d2c39a','#90bce8'),
                ('#5e8b6d','#8db185','#b0a077'))
        for y,band in enumerate(colors):
            for x,color in enumerate(band):
                p.fillRect(QRectF(4+x*7,5+y*6,6,5),QColor(color))
        p.setPen(QPen(QColor('#b9d1e5'),1.2)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(2,3,24,22),3,3)
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

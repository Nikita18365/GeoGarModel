"""Pick and edit raster control points before applying an affine transform."""
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QColor, QImage, QPen, QPixmap, QPainterPath
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGraphicsItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout)

from .raster_map import fit_affine


class RasterPreview(QGraphicsView):
    def __init__(self, image, add_point):
        super().__init__()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setScene(QGraphicsScene(self))
        height,width = image.shape[:2]
        picture = QImage(image.data,width,height,4*width,QImage.Format_RGBA8888)
        self.picture = self.scene().addPixmap(QPixmap.fromImage(picture))
        self.add_point = add_point
        self.setSceneRect(self.picture.boundingRect())
        self.viewport().setCursor(Qt.CrossCursor)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self._zoom = 1.
        self._fit_scale = 1.
        self._pan_pos = None

    def scene_position(self, viewport_pos):
        """Preserve fractional viewport coordinates, using the actual view transform."""
        inverse,ok = self.viewportTransform().inverted()
        if not ok:
            raise RuntimeError('Не удалось определить положение курсора.')
        return inverse.map(QPointF(viewport_pos))

    def _viewport_center(self):
        return QPointF(self.viewport().width()/2.,self.viewport().height()/2.)

    def _expand_scene(self):
        # At every zoom allow an entire half viewport beyond each edge.
        # Even a corner can be centred while the raster keeps its true
        # pixel coordinates. The margin is navigation space, not image data.
        scale = max(self.transform().m11(),1e-12)
        x = (self.viewport().width()/2.+24.)/scale
        y = (self.viewport().height()/2.+24.)/scale
        self.setSceneRect(self.picture.boundingRect().adjusted(-x,-y,x,y))

    def fit_image(self):
        """Reset the view, retaining a small visible margin on all sides."""
        bounds = self.picture.boundingRect()
        self._zoom = 1.
        self._fit_scale = min(max(1.,self.viewport().width()-32.)/bounds.width(),
                              max(1.,self.viewport().height()-32.)/bounds.height())
        self.resetTransform()
        self.scale(self._fit_scale,self._fit_scale)
        self._expand_scene()
        self.centerOn(bounds.center())

    def _max_zoom(self):
        # Previously 16 logical screen pixels per source pixel.
        # 100 gives 6.25 times the previous maximum magnification.
        return max(1.,100./max(self._fit_scale,1e-12))

    def resizeEvent(self, event):
        center = self.scene_position(self._viewport_center())
        zoom = self._zoom
        super().resizeEvent(event)
        self.fit_image()
        if zoom>1.:
            self._zoom = min(zoom,self._max_zoom())
            self.scale(self._zoom,self._zoom)
            self._expand_scene()
            self.centerOn(center)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            delta = event.pixelDelta().y()*3
        if not delta:
            event.ignore()
            return
        zoom = min(self._max_zoom(),max(1.,self._zoom*1.25**(max(-1200,min(1200,delta))/120.)))
        if zoom<=1.:
            self.fit_image()
        elif zoom!=self._zoom:
            pos = event.position() if hasattr(event,'position') else event.posF()
            anchor = self.scene_position(pos)
            factor = zoom/self._zoom
            self.scale(factor,factor)
            self._zoom = zoom
            self._expand_scene()
            center = self.scene_position(self._viewport_center())
            self.centerOn(center+anchor-self.scene_position(pos))
        event.accept()

    def mousePressEvent(self, event):
        if event.button()==Qt.MiddleButton:
            self._pan_pos = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        pos = self.scene_position(event.localPos())
        if event.button()==Qt.LeftButton and self._pan_pos is None and self.picture.boundingRect().contains(pos):
            # Continuous edge-based pixel coordinates are valid for the
            # affine transform. Do not snap to the centre of a pixel.
            self.add_point(pos.x(),pos.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_pos is not None:
            delta = self.scene_position(self._pan_pos)-self.scene_position(event.pos())
            self.centerOn(self.scene_position(self._viewport_center())+delta)
            self._pan_pos = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.MiddleButton and self._pan_pos is not None:
            self._pan_pos = None
            self.viewport().setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RasterPoints(QDialog):
    def __init__(self, image, crs, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Привязка растра')
        self.resize(1050,640)
        self.pixels = []
        self.markers = []
        layout = QVBoxLayout(self)
        hint = QLabel('Нажмите на изображение, затем введите координаты X и Y в таблице. '
                      'Нужны минимум три точки, расположенные не на одной прямой. '
                      'Колесо — масштабирование около курсора; зажатое колесо — перемещение. '
                      'Крестик устанавливается точно в месте щелчка, включая доли пикселя.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(QLabel('Исходная СК: '+crs.name))
        body = QHBoxLayout()
        layout.addLayout(body,1)
        self.preview = RasterPreview(image,self.add_point)
        image_panel = QVBoxLayout()
        body.addLayout(image_panel,3)
        image_panel.addWidget(self.preview,1)
        fit = QPushButton('Показать весь растр')
        fit.clicked.connect(self.preview.fit_image)
        image_panel.addWidget(fit)
        side = QVBoxLayout()
        body.addLayout(side,2)
        self.table = QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(['Пиксель X','Пиксель Y','Координата X','Координата Y'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        side.addWidget(self.table)
        remove = QPushButton('Удалить выбранную точку')
        remove.clicked.connect(self.remove_point)
        side.addWidget(remove)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton('Готово',QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.finish)
        layout.addWidget(buttons)

    def add_point(self,x,y):
        if (x,y) in self.pixels:
            self.table.setCurrentCell(self.pixels.index((x,y)),2)
            return
        row = len(self.pixels)
        self.pixels.append((x,y))
        self.table.insertRow(row)
        for column,value in enumerate((f'{x:.6f}',f'{y:.6f}','','')):
            item = QTableWidgetItem(value)
            if column<2:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row,column,item)
        # Draw a thin cross centred on the exact point, without covering
        # the target with a filled disc. Both marker and number stay legible.
        path = QPainterPath()
        path.moveTo(-6,0); path.lineTo(6,0)
        path.moveTo(0,-6); path.lineTo(0,6)
        path.addEllipse(QRectF(-4,-4,8,8))
        marker = self.preview.scene().addPath(path,QPen(QColor('#ff492f'),1.5))
        marker.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        marker.setZValue(10)
        marker.setPos(x,y)
        label = self.preview.scene().addSimpleText(str(row+1))
        label.setParentItem(marker)
        label.setBrush(QColor('#ff492f'))
        label.setPos(8,-16)
        self.markers.append((marker,label))
        self.table.setCurrentCell(row,2)
        self.table.editItem(self.table.item(row,2))

    def remove_point(self):
        row = self.table.currentRow()
        if row<0: return
        self.table.removeRow(row)
        self.pixels.pop(row)
        marker,_ = self.markers.pop(row)
        self.preview.scene().removeItem(marker)
        for index,(_,label) in enumerate(self.markers,1):
            label.setText(str(index))

    def finish(self):
        try:
            world = [[float(self.table.item(row,col).text().replace(',','.')) for col in (2,3)]
                     for row in range(len(self.pixels))]
        except ValueError:
            self.error.setText('Введите числовые координаты X и Y для каждой точки.')
            return
        try:
            self.transform = fit_affine(self.pixels,world)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.points = [list(pixel)+coords for pixel,coords in zip(self.pixels,world)]
        self.accept()

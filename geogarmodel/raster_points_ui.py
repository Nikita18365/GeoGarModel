"""Pick and edit raster control points before applying an affine transform."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPen, QPixmap
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
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self._zoom = 1.
        self._fit_scale = 1.
        self._pan_pos = None

    def fit_image(self):
        """Reset the view without modifying source pixels or control points."""
        self._zoom = 1.
        self.fitInView(self.picture,Qt.KeepAspectRatio)
        self._fit_scale = self.transform().m11()
        self.centerOn(self.picture.boundingRect().center())

    def _max_zoom(self):
        # Allow large scans to reach individual pixels (up to 16 screen
        # pixels per source pixel), without unbounded scene transforms.
        return max(1.,16./max(self._fit_scale,1e-12))

    def resizeEvent(self, event):
        center = self.mapToScene(self.viewport().rect().center())
        zoom = self._zoom
        super().resizeEvent(event)
        self.fit_image()
        if zoom>1.:
            self._zoom = min(zoom,self._max_zoom())
            self.scale(self._zoom,self._zoom)
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
            pos = event.pos()
            anchor = self.mapToScene(pos)
            factor = zoom/self._zoom
            self.scale(factor,factor)
            self._zoom = zoom
            center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(center+anchor-self.mapToScene(pos))
        event.accept()

    def mousePressEvent(self, event):
        if event.button()==Qt.MiddleButton:
            self._pan_pos = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        pos = self.mapToScene(event.pos())
        if event.button()==Qt.LeftButton and self._pan_pos is None and self.picture.boundingRect().contains(pos):
            # Scene coordinates always refer to the original raster pixels.
            self.add_point(int(pos.x())+.5,int(pos.y())+.5)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_pos is not None:
            delta = self.mapToScene(self._pan_pos)-self.mapToScene(event.pos())
            self.centerOn(self.mapToScene(self.viewport().rect().center())+delta)
            self._pan_pos = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.MiddleButton and self._pan_pos is not None:
            self._pan_pos = None
            self.viewport().unsetCursor()
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
                      'Колесо — масштабирование около курсора; зажатое колесо — перемещение.')
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
        for column,value in enumerate((f'{x:g}',f'{y:g}','','')):
            item = QTableWidgetItem(value)
            if column<2:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row,column,item)
        marker = self.preview.scene().addEllipse(-5,-5,10,10,QPen(QColor('white'),2),QColor('#e14a3b'))
        marker.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        marker.setPos(x,y)
        label = self.preview.scene().addSimpleText(str(row+1))
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        label.setBrush(QColor('#ff492f'))
        label.setPos(x,y)
        self.markers.append((marker,label))
        self.table.setCurrentCell(row,2)
        self.table.editItem(self.table.item(row,2))

    def remove_point(self):
        row = self.table.currentRow()
        if row<0: return
        self.table.removeRow(row)
        self.pixels.pop(row)
        for marker in self.markers.pop(row):
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

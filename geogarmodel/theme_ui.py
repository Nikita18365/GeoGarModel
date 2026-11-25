"""Live interface theme switching without rebuilding the project."""
from PyQt5.QtCore import QObject,QSettings,Qt
from PyQt5.QtGui import QColor,QIcon,QPixmap,QPainter,QBrush
from PyQt5.QtWidgets import QActionGroup,QApplication,QMdiArea,QDialog,QMainWindow
from .theme import (THEMES,DEFAULT_THEME,THEME_SETTINGS_KEY,theme_id,set_active_theme,
                    stylesheet,qt_palette,style_figure)
from .window_theme import apply_window_theme


def swatch(theme):
    pix=QPixmap(24,18); pix.fill(Qt.transparent)
    p=QPainter(pix)
    p.fillRect(0,0,24,18,QColor(theme.border))
    p.fillRect(1,1,22,16,QColor(theme.background))
    p.fillRect(3,3,8,12,QColor(theme.surface))
    p.fillRect(13,3,8,5,QColor(theme.accent))
    p.fillRect(13,10,8,5,QColor(theme.primary))
    p.end()
    return QIcon(pix)


class ThemeController(QObject):
    def __init__(self,window):
        super().__init__(window)
        self.window=window
        self.settings=QSettings('GeoGarModel','GeoGarModel')
        self.key=theme_id(self.settings.value(THEME_SETTINGS_KEY,DEFAULT_THEME,type=str))
        self.actions={}

    def initialize(self):
        t=set_active_theme(self.key)
        QApplication.instance().setPalette(qt_palette(t))
        self.window.setStyleSheet(stylesheet(self.key))

    def install_menu(self):
        self.menu=self.window.menuBar().addMenu('Темы')
        self.group=QActionGroup(self.menu); self.group.setExclusive(True)
        for key,t in THEMES.items():
            action=self.menu.addAction(swatch(t),t.name)
            action.setCheckable(True); action.setData(key)
            self.group.addAction(action)
            action.setChecked(key==self.key)
            action.triggered.connect(lambda checked,k=key:self.select(k) if checked else None)
            self.actions[key]=action

    def select(self,key):
        self.key=theme_id(key)
        self.initialize()
        self.refresh()
        if self.key in self.actions: self.actions[self.key].setChecked(True)
        self.settings.setValue(THEME_SETTINGS_KEY,self.key)
        self.settings.sync()

    def refresh(self):
        from .theme import current_theme
        from .map_export_ui import format_icon
        import pyqtgraph as pg
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        t=current_theme()
        for area in self.window.findChildren(QMdiArea):
            area.setBackground(QBrush(QColor(t.surface)))
        for plot in self.window.findChildren(pg.PlotWidget):
            plot.setBackground(t.panel)
            item=plot.getPlotItem()
            for name in ('left','right','top','bottom'):
                axis=item.getAxis(name)
                axis.setPen(t.border); axis.setTextPen(t.text)
                if axis.labelText:
                    options=dict(axis.labelStyle); options['color']=t.text
                    axis.setLabel(text=axis.labelText,units=axis.labelUnits,**options)
            if item.titleLabel.text:
                options=dict(item.titleLabel.opts); options['color']=t.text
                item.setTitle(item.titleLabel.text,**options)
            if item.legend is not None:
                item.legend.setLabelTextColor(t.text)
            bar=getattr(plot,'bar',None)
            if isinstance(bar,pg.ColorBarItem):
                axis=bar.getAxis('right')
                axis.setPen(t.border); axis.setTextPen(t.text)
        for canvas in self.window.findChildren(FigureCanvasQTAgg):
            if getattr(canvas.figure,'_geogar_theme_figure',False):
                style_figure(canvas.figure); canvas.draw_idle()
        for fmt,button in getattr(self.window,'map_export_buttons',{}).items():
            button.setIcon(format_icon(fmt))
        chat=getattr(self.window,'manual_assistant',None)
        if chat is not None:
            bar=chat.history.verticalScrollBar()
            position=bar.value()
            chat.load_chat()
            bar.setValue(min(position,bar.maximum()))
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget,(QMainWindow,QDialog)) and (widget is self.window or self.window.isAncestorOf(widget)):
                if widget.isVisible(): apply_window_theme(widget)
        self.window.update()

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure
from geogarmodel.theme import (THEMES,DEFAULT_THEME,THEME_SETTINGS_KEY,
    stylesheet,theme_id,qt_palette,set_active_theme,current_theme,style_map_figure)
from geogarmodel.app import MainWindow


def test_theme_catalog():
    assert [t.name for t in THEMES.values()]==['Midnight Blue','Arctic Light',
        'Geological Atlas','Amethyst','Taiga','Copper Core']
    assert theme_id('invalid')==DEFAULT_THEME
    assert len({stylesheet(k) for k in THEMES})==6
    for key,t in THEMES.items():
        assert t.background in stylesheet(key)
        assert t.text in stylesheet(key)
        assert qt_palette(t).windowText().color().name()==t.text


def test_switch_theme_preserves_map_and_settings(qt_application_lifetime):
    import numpy as np
    from geogarmodel.grids import Grid
    settings=QSettings('GeoGarModel','GeoGarModel')
    old=settings.value(THEME_SETTINGS_KEY)
    previous_palette=QApplication.palette()
    previous_key=next(k for k,t in THEMES.items() if t==current_theme())
    w=MainWindow()
    try:
        assert w.theme_controller.menu.title()=='Темы'
        for key,t in THEMES.items():
            w.theme_controller.actions[key].trigger()
            assert w.theme_controller.key==key
            assert w.theme_controller.actions[key].isChecked()
            assert settings.value(THEME_SETTINGS_KEY)==key
            assert QApplication.palette().windowText().color().name()==t.text
        figure=Figure()
        style_map_figure(figure)
        assert figure.get_facecolor()==(1.,1.,1.,1.)
        w.close()
        w=MainWindow()
        assert w.theme_controller.key=='copper'
    finally:
        w.close()
        if old is None: settings.remove(THEME_SETTINGS_KEY)
        else: settings.setValue(THEME_SETTINGS_KEY,old)
        settings.sync()
        set_active_theme(previous_key)
        QApplication.instance().setPalette(previous_palette)

"""Application-wide palette and spacing, independent of numerical code."""
import re
from dataclasses import dataclass

STYLE = """
QMainWindow, QDialog, QWidget { background: #0b1220; color: #dbe7f6; font-size: 13px; }
QWidget#mainTabTools { background: transparent; }
QLabel#muted { color: #93a9c4; }
QLabel#metricCard { background: #14263b; border: 1px solid #243e59; border-radius: 9px; padding: 6px 9px; font-size: 15px; font-weight: 600; }
QLabel { background: transparent; }
QTabWidget::pane { border: 1px solid #24364e; border-radius: 8px; }
QTabBar::tab { background: #101c2f; color: #a9bbd3; padding: 9px 10px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #38bdf8; border-bottom: 2px solid #38bdf8; background: #15263c; }
QTabBar::tab:hover { background: #1b3049; }
QPushButton#workflowStep { background: #14263b; color: #cbdff5; border: 1px solid #304965; border-radius: 6px; padding: 8px 3px; text-align: center; font-weight: 600; }
QPushButton#workflowStep:checked { background: #1b394c; color: #38bdf8; border-color: #38bdf8; }
QWidget#inversionParameters QComboBox:disabled, QWidget#inversionParameters QCheckBox:disabled { color: #cbdff5; }
QPushButton, QToolButton { background: #182b43; border: 1px solid #2a4360; border-radius: 6px; padding: 7px 11px; }
QWidget#mainTabTools QPushButton, QWidget#mainTabTools QToolButton { padding: 3px 8px; }
QWidget#mainTabTools QComboBox { padding: 3px 6px; }
QPushButton:hover, QToolButton:hover { background: #23405d; border-color: #38bdf8; }
QPushButton:disabled { color: #586d86; background: #101c2f; }
QPushButton#primary { background: #147d91; border-color: #2dd4bf; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #1995a8; }
QPushButton#stop { color: #fda4af; border-color: #713343; }
QPushButton#stop:disabled, QPushButton#primary:disabled { color: #586d86; background: #101c2f; border-color: #24364e; }
QPushButton#disclosure { background: transparent; border: 0; border-bottom: 1px solid #24364e; text-align: left; color: #a7c5e7; padding: 8px 3px; }
QPushButton#disclosure:hover { color: #38bdf8; background: #101c2f; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #101c2f; border: 1px solid #2a4360; border-radius: 5px; padding: 6px; selection-background-color: #165c7a; }
QLineEdit:focus, QComboBox:focus { border-color: #38bdf8; }
QComboBox QAbstractItemView { background: #14263b; selection-background-color: #165c7a; }
QListWidget, QTableWidget { background: #101c2f; alternate-background-color: #142238; border: 1px solid #24364e; border-radius: 5px; selection-background-color: #165c7a; }
QHeaderView::section { background: #1a2d45; color: #b9cde5; padding: 7px; border: 0; }
QGroupBox { font-weight: 600; border: 1px solid #24364e; border-radius: 8px; margin-top: 12px; padding: 18px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #a7c5e7; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #101c2f; width: 10px; }
QScrollBar::handle:vertical { background: #304965; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar { background: #101c2f; border: 1px solid #24364e; border-radius: 5px; text-align: center; min-height: 18px; }
QProgressBar::chunk { background: #147d91; border-radius: 4px; }
QStatusBar { background: #101c2f; color: #93a9c4; }
QMenuBar, QMenu { background: #101c2f; color: #dbe7f6; }
QMenu::item:selected { background: #23405d; }
QSplitter::handle { background: #1a2d45; width: 5px; }
QSplitter#parameterSplitter::handle:horizontal { width: 22px; background: #20344d; border-left: 1px solid #65829f; }
QSplitter#parameterSplitter::handle:horizontal:hover { border-left-color: #38bdf8; }
QToolTip { background: #203b53; color: white; border: 1px solid #38bdf8; padding: 5px; }
"""


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    panel: str
    surface: str
    border: str
    text: str
    muted: str
    accent: str
    primary: str
    light: bool = False


THEMES = {
    'midnight': Theme('Midnight Blue','#0b1220','#101c2f','#182b43','#304965','#dbe7f6','#93a9c4','#38bdf8','#147d91'),
    'arctic': Theme('Arctic Light','#edf2f7','#ffffff','#e4ebf4','#a6b9cf','#23364b','#536a81','#236bc4','#276ec0',True),
    'atlas': Theme('Geological Atlas','#e5dfd1','#f7f2e7','#e8e0cc','#b0a187','#3c4436','#686452','#586b31','#677a49',True),
    'amethyst': Theme('Amethyst','#151122','#221a35','#32254b','#4e396b','#ece1fa','#b1a0cc','#bf9bff','#7045ad'),
    'taiga': Theme('Taiga','#101e19','#182e25','#254336','#3a5c47','#deeee0','#98b6a0','#a0d781','#407c50'),
    'copper': Theme('Copper Core','#201715','#30221d','#443028','#684737','#f5e2cf','#c6a08b','#f1ac70','#9b5632'),
}
DEFAULT_THEME = 'midnight'
THEME_SETTINGS_KEY = 'appearance/theme'
_active_theme = DEFAULT_THEME


def theme_id(value):
    return value if isinstance(value,str) and value in THEMES else DEFAULT_THEME


def current_theme():
    return THEMES[_active_theme]


def set_active_theme(value):
    global _active_theme
    _active_theme=theme_id(value)
    return current_theme()


def blend(a,b,fraction):
    return '#'+''.join(f'{round(int(a[i:i+2],16)*(1-fraction)+int(b[i:i+2],16)*fraction):02x}'
                       for i in (1,3,5))


def contrast_text(background):
    def linear(v):
        v=v/255
        return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
    luminance=sum(w*linear(int(background[i:i+2],16)) for i,w in zip((1,3,5),(.2126,.7152,.0722)))
    return '#000000' if luminance>.179 else '#ffffff'


def theme_colors(theme=None):
    t=theme or current_theme()
    return dict(hover=blend(t.surface,t.accent,.15),
                selection='#165c7a' if t.name=='Midnight Blue' else blend(t.panel,t.accent,.20 if t.light else .28),
                disabled=blend(t.muted,t.panel,.36),
                danger='#a51d36' if t.light else '#fda4af',
                success='#27663d' if t.light else '#86d6a0',
                gold='#8a5a09' if t.light else '#edc775')


def stylesheet(value):
    key=theme_id(value); t=THEMES[key]; c=theme_colors(t)
    replacements={
        '#0b1220':t.background,'#101c2f':t.panel,'#182b43':t.surface,
        '#dbe7f6':t.text,'#93a9c4':t.muted,'#304965':t.border,
        '#38bdf8':t.accent,'#147d91':t.primary,
        '#14263b':t.surface,'#243e59':t.border,'#24364e':t.border,
        '#a9bbd3':t.muted,'#15263c':c['selection'],'#1b3049':c['hover'],
        '#cbdff5':t.text,'#1b394c':c['selection'],'#2a4360':t.border,
        '#23405d':c['hover'],'#586d86':c['disabled'],'#2dd4bf':t.accent,
        '#1995a8':blend(t.primary,t.accent,.18),'#fda4af':c['danger'],
        '#713343':blend(t.border,c['danger'],.45),'#a7c5e7':t.text,
        '#165c7a':c['selection'],'#142238':blend(t.panel,t.surface,.5),
        '#1a2d45':t.surface,'#b9cde5':t.text,'#20344d':t.surface,
        '#65829f':t.border,'#203b53':t.surface,
    }
    qss=STYLE if key==DEFAULT_THEME else re.sub(r'#[0-9a-fA-F]{6}',lambda m:replacements.get(m[0].lower(),m[0]),STYLE)
    return qss+f"""
QTextBrowser, QTextEdit, QTreeView, QTableView {{ background: {t.panel}; color: {t.text}; selection-background-color: {c['selection']}; selection-color: {t.text}; }}
QAbstractItemView::item:selected {{ background: {c['selection']}; color: {t.text}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{ selection-color: {t.text}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {c['hover']}; color: {t.text}; }}
QMenu::item:disabled {{ color: {c['disabled']}; }}
QToolTip {{ color: {t.text}; }}
QPushButton#primary {{ color: {contrast_text(t.primary)}; }}
QPushButton#primary:hover {{ color: {contrast_text(blend(t.primary,t.accent,.18))}; }}
QPushButton#primary:disabled {{ color: {c['disabled']}; }}
QLabel#gridCalculatorError {{ color: {c['danger']}; }}
QListWidget#modelList::item {{ padding: 4px; }}
QLineEdit#layerNameEditor {{ padding: 2px 4px; border: 1px solid {t.accent}; }}
QToolButton#petDismiss {{ padding: 0; border: 1px solid {t.border}; background: {t.surface}; border-radius: 10px; }}
QLabel#petGreeting {{ color: {t.text}; background: {t.surface}; border: 1px solid {t.border}; border-radius: 7px; padding: 8px; }}
"""


def qt_palette(theme=None):
    from PyQt5.QtGui import QColor,QPalette
    t=theme or current_theme(); c=theme_colors(t)
    p=QPalette()
    for role,color in ((QPalette.Window,t.background),(QPalette.WindowText,t.text),
                       (QPalette.Base,t.panel),(QPalette.AlternateBase,t.surface),
                       (QPalette.Text,t.text),(QPalette.Button,t.surface),(QPalette.ButtonText,t.text),
                       (QPalette.Highlight,c['selection']),(QPalette.HighlightedText,t.text),
                       (QPalette.ToolTipBase,t.surface),(QPalette.ToolTipText,t.text),
                       (QPalette.Link,t.accent),(QPalette.LinkVisited,t.accent),
                       (QPalette.Mid,t.border),(QPalette.Dark,t.border)):
        p.setColor(role,QColor(color))
    for role in (QPalette.Text,QPalette.WindowText,QPalette.ButtonText):
        p.setColor(QPalette.Disabled,role,QColor(c['disabled']))
    return p


def style_figure(figure):
    figure._geogar_theme_figure=True
    t=current_theme()
    figure.set_facecolor(t.panel)
    for text in figure.texts:
        text.set_color(t.text)
    for ax in figure.axes:
        ax.set_facecolor(t.panel)
        ax.tick_params(colors=t.muted,labelsize=9)
        ax.xaxis.label.set_color(t.muted)
        ax.yaxis.label.set_color(t.muted)
        ax.title.set_color(t.text)
        for spine in ax.spines.values():
            spine.set_edgecolor(t.border)


def style_map_figure(figure):
    """White publication sheet for maps, independent of the application theme."""
    figure._geogar_theme_figure=False
    figure.set_facecolor('white')
    for ax in figure.axes:
        ax.set_facecolor('white'); ax.tick_params(colors='#222222',labelsize=10)
        ax.xaxis.label.set_color('#222222'); ax.yaxis.label.set_color('#222222')
        ax.xaxis.get_offset_text().set_color('#222222'); ax.yaxis.get_offset_text().set_color('#222222')
        ax.title.set_color('#111111')
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')

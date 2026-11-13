"""Application-wide palette and spacing, independent of numerical code."""
STYLE = """
QMainWindow, QDialog, QWidget { background: #0b1220; color: #dbe7f6; font-size: 13px; }
QWidget#header { background: #101c2f; border-bottom: 1px solid #24364e; }
QLabel#brand { font-size: 23px; font-weight: 700; color: #f1f7ff; }
QLabel#muted { color: #93a9c4; }
QLabel#logo { background: #123c49; color: #2dd4bf; font-size: 24px; font-weight: 700; border-radius: 10px; }
QLabel#metricCard { background: #14263b; border: 1px solid #243e59; border-radius: 9px; padding: 10px 15px; font-size: 17px; font-weight: 600; }
QLabel { background: transparent; }
QTabWidget::pane { border: 1px solid #24364e; border-radius: 8px; }
QTabBar::tab { background: #101c2f; color: #93a9c4; padding: 11px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #38bdf8; border-bottom: 2px solid #38bdf8; background: #15263c; }
QTabBar::tab:hover { background: #1b3049; }
QToolBox::tab { background: #14263b; color: #a7c5e7; border: 1px solid #24364e; border-radius: 5px; padding: 9px; text-align: left; }
QToolBox::tab:selected { background: #1b394c; color: #38bdf8; font-weight: 600; }
QPushButton, QToolButton { background: #182b43; border: 1px solid #2a4360; border-radius: 6px; padding: 7px 11px; }
QPushButton:hover, QToolButton:hover { background: #23405d; border-color: #38bdf8; }
QPushButton:disabled { color: #586d86; background: #101c2f; }
QPushButton#primary { background: #147d91; border-color: #2dd4bf; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #1995a8; }
QPushButton#stop { color: #fda4af; border-color: #713343; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #101c2f; border: 1px solid #2a4360; border-radius: 5px; padding: 6px; selection-background-color: #165c7a; }
QLineEdit:focus, QComboBox:focus { border-color: #38bdf8; }
QComboBox QAbstractItemView { background: #14263b; selection-background-color: #165c7a; }
QListWidget, QTableWidget { background: #101c2f; alternate-background-color: #142238; border: 1px solid #24364e; border-radius: 5px; selection-background-color: #165c7a; }
QHeaderView::section { background: #1a2d45; color: #b9cde5; padding: 7px; border: 0; }
QGroupBox { font-weight: 600; border: 1px solid #24364e; border-radius: 8px; margin-top: 12px; padding-top: 14px; }
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
QToolTip { background: #203b53; color: white; border: 1px solid #38bdf8; padding: 5px; }
"""


def style_figure(figure):
    figure.set_facecolor("#101c2f")
    for ax in figure.axes:
        ax.set_facecolor("#101c2f")
        ax.tick_params(colors="#a9bbd3", labelsize=9)
        ax.xaxis.label.set_color("#a9bbd3")
        ax.yaxis.label.set_color("#a9bbd3")
        ax.title.set_color("#dbe7f6")
        for spine in ax.spines.values():
            spine.set_edgecolor("#304965")

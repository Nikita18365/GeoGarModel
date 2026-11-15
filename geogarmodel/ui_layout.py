"""Shared presentation helpers. These widgets never change calculation settings."""
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QPushButton,QScrollArea,QFormLayout,
    QSizePolicy,QLabel,QGroupBox)


class Disclosure(QWidget):
    """A readable header reveals supplementary controls without disabling them."""
    def __init__(self,title,content,expanded=False,parent=None):
        super().__init__(parent)
        rows=QVBoxLayout(self); rows.setContentsMargins(0,0,0,0); rows.setSpacing(6)
        self.toggle=QPushButton(); self.toggle.setObjectName('disclosure')
        self.toggle.setProperty('workflowNavigation',True)
        self.toggle.setCheckable(True); self.toggle.setChecked(expanded)
        self.toggle.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Minimum)
        self.content=content; rows.addWidget(self.toggle); rows.addWidget(content)
        def show(opened):
            self.toggle.setText(('▾  ' if opened else '▸  ')+title)
            content.setVisible(opened)
        self.toggle.toggled.connect(show); show(expanded)


class SummaryLabel(QLabel):
    """Single-line summary; full original text remains in tooltip and text()."""
    def setText(self,text):
        super().setText(text); self.setToolTip(text); self.setAccessibleName(text)

    def sizeHint(self): return QSize(260,self.fontMetrics().height()+6)
    def minimumSizeHint(self): return QSize(50,self.fontMetrics().height()+6)

    def paintEvent(self,event):
        painter=QPainter(self); painter.setPen(self.palette().windowText().color())
        text=self.fontMetrics().elidedText(self.text().replace('\n',' · '),Qt.ElideRight,self.contentsRect().width())
        painter.drawText(self.contentsRect(),Qt.AlignLeft | Qt.AlignVCenter,text)


def details(layout,widget,title,expanded=False):
    index=layout.indexOf(widget)
    layout.takeAt(index)
    section=Disclosure(title,widget,expanded)
    layout.insertWidget(index,section)
    return section


def form_details(form,widgets,title):
    """Move selected rows, retaining their original widgets and connections."""
    content=QWidget(); inner=QFormLayout(content)
    for widget in widgets:
        row=form.takeRow(widget)
        label=row.labelItem.widget() if row.labelItem else None
        field=row.fieldItem.widget() if row.fieldItem else None
        if label is not None: inner.addRow(label,field)
        elif field is not None: inner.addRow(field)
    inner.setRowWrapPolicy(QFormLayout.WrapLongRows)
    inner.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    section=Disclosure(title,content); form.addRow(section)
    return section


def workflow_sections(layout,sections):
    from .inversion_workflow_ui import Workflow
    workflow=Workflow()
    for title,widgets in sections:
        page=QWidget(); rows=QVBoxLayout(page); rows.setContentsMargins(0,12,0,0)
        for widget in widgets:
            layout.removeWidget(widget); rows.addWidget(widget)
        rows.addStretch()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        workflow.addItem(scroll,title)
    workflow.finish(); layout.addWidget(workflow,1)
    return workflow


def quiet_button(button,tooltip=None):
    button.setToolTip(tooltip or button.text())
    shorter={'Загрузить воксель / UBC / VTK / центры…':'Загрузить модель…',
             'Использовать результат последнего моделирования':'Взять предыдущий результат',
             'Загрузить отдельно JE, JN, JD…':'Загрузить компоненты JE / JN / JD…'}
    if button.text() in shorter: button.setText(shorter[button.text()])
    button.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Minimum)
    button.setMinimumHeight(34)


def prepare_forms(root):
    for group in root.findChildren(QGroupBox):
        form=group.layout()
        if isinstance(form,QFormLayout):
            # Stacked label/field rows cannot collide at high DPI or in narrow panels.
            form.setRowWrapPolicy(QFormLayout.WrapAllRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setVerticalSpacing(8)
    for label in root.findChildren(QLabel):
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)

"""Present observation, initial model and solver forms as three explicit steps."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QStackedWidget,QScrollArea,QPushButton,QSizePolicy


class Workflow(QWidget):
    """Explicit navigation buttons avoid native QToolBox title clipping on Windows."""
    def __init__(self):
        super().__init__()
        self.rows=QVBoxLayout(self); self.rows.setContentsMargins(0,0,0,0)
        self.navigation=QHBoxLayout(); self.navigation.setSpacing(4); self.rows.addLayout(self.navigation)
        self.buttons=[]; self.pages=QStackedWidget()

    def addItem(self,page,title):
        index=self.pages.addWidget(page)
        button=QPushButton(title); button.setObjectName('workflowStep')
        button.setToolTip(title)
        button.setProperty('workflowNavigation',True); button.setCheckable(True)
        button.setMinimumHeight(button.fontMetrics().height()+24)
        button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Minimum)
        button.clicked.connect(lambda checked=False,i=index:self.setCurrentIndex(i))
        button.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Minimum)
        self.buttons.append(button); self.navigation.addWidget(button)

    def finish(self):
        self.rows.addWidget(self.pages,1); self.setCurrentIndex(0)

    def setCurrentIndex(self,index):
        self.pages.setCurrentIndex(index)
        for i,button in enumerate(self.buttons): button.setChecked(i==index)

    def currentIndex(self): return self.pages.currentIndex()
    def count(self): return self.pages.count()
    def widget(self,index): return self.pages.widget(index)


def arrange_workflow(layout,initial,mesh,properties,magnetic,solver,compute):
    from .ui_layout import Disclosure
    # Keep physics selector and explanation visible across all steps.
    widgets=[]
    while layout.count()>2:
        widgets.append(layout.takeAt(2).widget())
    models=[initial,mesh,properties]; settings=[magnetic,solver,compute]
    observations=[w for w in widgets if w not in models+settings]
    extra=QWidget(); extra_rows=QVBoxLayout(extra)
    for widget in [observations[2],observations[4],observations[5]]:
        extra_rows.addWidget(widget)
    observations=[observations[0],observations[1],observations[3],Disclosure('Другие источники и примеры',extra),*observations[6:]]
    workflow=Workflow()
    for index,(title,items) in enumerate([('1 · Поле',observations),('2 · Модель',models),('3 · Расчёт',settings)]):
        page=QWidget(); rows=QVBoxLayout(page)
        for widget in items: rows.addWidget(widget)
        if index<2:
            button=QPushButton('Далее → '+('Начальная модель' if index==0 else 'Настройки расчёта'))
            button.setProperty('workflowNavigation',True)
            button.clicked.connect(lambda checked=False,i=index:workflow.setCurrentIndex(i+1)); rows.addWidget(button)
        rows.addStretch()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        workflow.addItem(scroll,title)
    workflow.finish()
    layout.addWidget(workflow,1)
    return workflow

"""Present observation, initial model and solver forms as three explicit steps."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QStackedWidget,QScrollArea,QPushButton,QSizePolicy


class Workflow(QWidget):
    """Explicit navigation buttons avoid native QToolBox title clipping on Windows."""
    def __init__(self):
        super().__init__()
        self.rows=QVBoxLayout(self); self.rows.setContentsMargins(0,0,0,0)
        self.buttons=[]; self.pages=QStackedWidget()

    def addItem(self,page,title):
        index=self.pages.addWidget(page)
        button=QPushButton(title); button.setObjectName('workflowStep')
        button.setProperty('workflowNavigation',True); button.setCheckable(True)
        button.setMinimumHeight(button.fontMetrics().height()+24)
        button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Minimum)
        button.clicked.connect(lambda checked=False,i=index:self.setCurrentIndex(i))
        self.buttons.append(button); self.rows.addWidget(button)

    def finish(self):
        self.rows.addWidget(self.pages,1); self.setCurrentIndex(0)

    def setCurrentIndex(self,index):
        self.pages.setCurrentIndex(index)
        for i,button in enumerate(self.buttons): button.setChecked(i==index)

    def currentIndex(self): return self.pages.currentIndex()
    def count(self): return self.pages.count()
    def widget(self,index): return self.pages.widget(index)


def arrange_workflow(layout,initial,mesh,properties,magnetic,solver,compute):
    # Keep physics selector and explanation visible across all steps.
    widgets=[]
    while layout.count()>2:
        widgets.append(layout.takeAt(2).widget())
    models=[initial,mesh,properties]; settings=[magnetic,solver,compute]
    observations=[w for w in widgets if w not in models+settings]
    workflow=Workflow()
    for index,(title,items) in enumerate([('1 · Наблюдённое поле',observations),('2 · Начальная модель и сетка',models),('3 · Настройки расчёта',settings)]):
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

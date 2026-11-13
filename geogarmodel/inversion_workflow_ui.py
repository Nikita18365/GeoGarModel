"""Present observation, initial model and solver forms as three explicit steps."""
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QToolBox,QScrollArea,QPushButton


def arrange_workflow(layout,initial,mesh,properties,magnetic,solver,compute):
    # Keep physics selector and explanation visible across all steps.
    widgets=[]
    while layout.count()>2:
        widgets.append(layout.takeAt(2).widget())
    models=[initial,mesh,properties]; settings=[magnetic,solver,compute]
    observations=[w for w in widgets if w not in models+settings]
    workflow=QToolBox()
    for index,(title,items) in enumerate([('1 · Наблюдённое поле',observations),('2 · Начальная модель и сетка',models),('3 · Настройки расчёта',settings)]):
        page=QWidget(); rows=QVBoxLayout(page)
        for widget in items: rows.addWidget(widget)
        if index<2:
            button=QPushButton('Далее → '+('Начальная модель' if index==0 else 'Настройки расчёта'))
            button.clicked.connect(lambda checked=False,i=index:workflow.setCurrentIndex(i+1)); rows.addWidget(button)
        rows.addStretch()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        workflow.addItem(scroll,title)
    layout.addWidget(workflow,1)
    return workflow

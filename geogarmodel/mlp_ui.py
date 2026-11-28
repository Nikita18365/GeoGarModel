"""MLP architecture editor with explicit ordered layers and synchronized JSON."""
import json
from copy import deepcopy
from pathlib import Path
from PyQt5.QtCore import QSignalBlocker,Qt
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,
    QTableWidget,QTableWidgetItem,QHeaderView,QComboBox,QPushButton,QPlainTextEdit,
    QDialogButtonBox,QFileDialog,QMessageBox,QTabWidget,QWidget,QSpinBox,QDoubleSpinBox,
    QCheckBox,QLineEdit,QStyledItemDelegate,QInputDialog)
from .mlp_config import defaults,validate_config,ACTIVATIONS,OPTIMIZERS,LOSSES,TRAIN_METRICS,EVAL_METRICS,NORMALIZATION


class ActivationDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        editor=QComboBox(parent)
        editor.addItems(ACTIVATIONS)
        return editor

    def setEditorData(self,editor,index):
        editor.setCurrentText(index.data(Qt.EditRole))

    def setModelData(self,editor,model,index):
        model.setData(index,editor.currentText(),Qt.EditRole)


class MLPDialog(QDialog):
    def __init__(self,seed,overrides,parent=None):
        super().__init__(parent)
        self.seed=seed
        self.parameters=None
        self.setWindowTitle("MLP · архитектура и обучение")
        self.resize(1240,780)
        self.setMinimumSize(680,500)
        layout=QVBoxLayout(self)
        note=QLabel("Вход определяется числом признаков. Слои выполняются сверху вниз. Выход — один линейный нейрон. Новые Dense линейные; добавляйте Activation отдельно. Старые встроенные активации сохранены. L1/L2 — только веса Dense.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tabs=QTabWidget()
        layout.addWidget(self.tabs,1)
        page=QWidget()
        arch=QVBoxLayout(page)
        self.layers=QTableWidget(0,7)
        self.layers.setHorizontalHeaderLabels(["Тип","Нейроны","Активация","Dropout rate","L1","L2","Параметры норм."])
        self.layers.setItemDelegateForColumn(2,ActivationDelegate(self.layers))
        self.layers.verticalHeader().setDefaultSectionSize(46)
        header=self.layers.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for col,width in enumerate((145,90,125,115,90,90,340)):
            self.layers.setColumnWidth(col,width)
        header.setStretchLastSection(True)
        self.layers.setWordWrap(False)
        self.layers.setSelectionBehavior(QTableWidget.SelectRows)
        self.layers.setSelectionMode(QTableWidget.SingleSelection)
        arch.addWidget(self.layers,1)
        buttons=QHBoxLayout()
        for title,action in (("Dense",lambda:self.add_layer("Dense")),("Activation",lambda:self.add_layer("Activation")),
                             ("Dropout",lambda:self.add_layer("Dropout")),("BatchNorm",lambda:self.add_layer("BatchNorm")),
                             ("LayerNorm",lambda:self.add_layer("LayerNorm")),("↑",lambda:self.move(-1)),
                             ("↓",lambda:self.move(1)),("Удалить",self.remove)):
            button=QPushButton(title)
            button.clicked.connect(action)
            buttons.addWidget(button)
        arch.addLayout(buttons)
        norm_button=QPushButton("Настроить выбранный слой нормализации…")
        norm_button.clicked.connect(self.edit_normalization)
        arch.addWidget(norm_button)
        norm_hint=QLabel("BatchNorm: batch size ≥ 2; одиночный хвост объединяется с предыдущим батчем. Прогноз использует статистики обучения. Для изменения eps / momentum / affine выберите строку нормализации.")
        norm_hint.setWordWrap(True)
        arch.addWidget(norm_hint)
        output=QFormLayout()
        self.output_l1=QLineEdit()
        self.output_l2=QLineEdit()
        output.addRow("L1 выходного слоя",self.output_l1)
        output.addRow("L2 выходного слоя",self.output_l2)
        arch.addLayout(output)
        self.tabs.addTab(page,"Архитектура")
        page=QWidget()
        form=QFormLayout(page)
        self.epochs=QSpinBox(); self.epochs.setRange(1,1000000)
        self.batch_size=QSpinBox(); self.batch_size.setRange(1,1000000)
        self.learning_rate=QLineEdit()
        self.gradient_clip=QLineEdit()
        self.random_state=QSpinBox(); self.random_state.setRange(0,2147483647)
        self.shuffle=QCheckBox("Перемешивать обучающие мини-батчи")
        self.optimizer=QComboBox(); self.optimizer.addItems(OPTIMIZERS)
        self.loss=QComboBox(); self.loss.addItems(LOSSES)
        self.optimizer_params=QPlainTextEdit(); self.optimizer_params.setMaximumHeight(105)
        self.loss_params=QPlainTextEdit(); self.loss_params.setMaximumHeight(80)
        for label,widget in (("Эпохи",self.epochs),("Batch size",self.batch_size),
                ("Learning rate",self.learning_rate),("Gradient clip (0 — выкл.)",self.gradient_clip),
                ("Seed",self.random_state),("",self.shuffle),("Оптимизатор",self.optimizer),
                ("Параметры оптимизатора (JSON)",self.optimizer_params),("Функция потерь",self.loss),
                ("Параметры loss (JSON)",self.loss_params)):
            form.addRow(label,widget)
        self.optimizer.currentTextChanged.connect(lambda name:self.optimizer_params.setPlainText(json.dumps(OPTIMIZERS[name],indent=2)))
        self.loss.currentTextChanged.connect(lambda name:self.loss_params.setPlainText(json.dumps(LOSSES[name],indent=2)))
        self.tabs.addTab(page,"Обучение")
        page=QWidget()
        metrics=QVBoxLayout(page)
        metrics.addWidget(QLabel("Метрики обучения: по всей обучающей выборке в eval-режиме после эпохи, на нормированной цели."))
        self.train_checks={name:QCheckBox(name) for name in TRAIN_METRICS}
        for box in self.train_checks.values(): metrics.addWidget(box)
        metrics.addWidget(QLabel("Контроль: стандартные RMSE, MAE, MSE и R² сохраняются всегда. Выберите дополнительные метрики."))
        self.eval_checks={name:QCheckBox(name) for name in EVAL_METRICS}
        for name,box in self.eval_checks.items():
            metrics.addWidget(box)
            if name in ("RMSE","MAE","MSE","R2"):
                box.setChecked(True); box.setEnabled(False)
        hint=QLabel("Контрольные метрики вычисляются в исходных единицах глубины. Optuna пока минимизирует RMSE независимо от loss.")
        hint.setWordWrap(True); metrics.addWidget(hint); metrics.addStretch()
        self.tabs.addTab(page,"Метрики")
        page=QWidget()
        jl=QVBoxLayout(page)
        self.editor=QPlainTextEdit()
        jl.addWidget(self.editor)
        row=QHBoxLayout()
        for title,action in (("Сформировать JSON из конструктора",self.to_json),("Применить JSON к конструктору",self.from_json)):
            b=QPushButton(title); b.clicked.connect(action); row.addWidget(b)
        jl.addLayout(row)
        self.tabs.addTab(page,"JSON")
        row=QHBoxLayout()
        for title,action in (("Восстановить default",lambda:self.load(defaults(self.seed))),("Экспорт JSON…",self.export)):
            b=QPushButton(title); b.clicked.connect(action); row.addWidget(b)
        layout.addLayout(row)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Зафиксировать для расчёта")
        buttons.accepted.connect(self.apply); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.load(validate_config(overrides,seed))
        self._previous_tab=0
        self.tabs.currentChanged.connect(self.sync_tab)

    def append_layer(self,spec):
        row=self.layers.rowCount(); self.layers.insertRow(row)
        kind=spec["type"]
        options=json.dumps({key:spec.get(key,value) for key,value in NORMALIZATION[kind].items()},ensure_ascii=False) if kind in NORMALIZATION else ""
        values=[kind,spec.get("units",""),spec.get("activation",""),spec.get("rate",""),spec.get("l1",""),spec.get("l2",""),options]
        enabled={"Dense":{1,2,4,5},"Activation":{2},"Dropout":{3},"LayerNorm":{6},"BatchNorm":{6}}[kind]
        for col,value in enumerate(values):
            item=QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            if col not in enabled: item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.layers.setItem(row,col,item)
        self.layers.setCurrentCell(row,0)

    def add_layer(self,kind):
        spec={"Dense":dict(type=kind,units=64,activation="linear",l1=0.,l2=0.),
              "Activation":dict(type=kind,activation="relu"),"Dropout":dict(type=kind,rate=.1)}
        self.append_layer(dict(type=kind,**NORMALIZATION[kind]) if kind in NORMALIZATION else spec[kind])

    def edit_normalization(self):
        row=self.layers.currentRow()
        if row<0 or self.layers.item(row,0).text() not in NORMALIZATION:
            QMessageBox.information(self,"MLP","Выберите строку BatchNorm или LayerNorm.")
            return
        kind=self.layers.item(row,0).text()
        cell=self.layers.item(row,6)
        try:
            current=json.loads(cell.text())
            initial=json.dumps(current,indent=2,ensure_ascii=False)
        except (ValueError,TypeError):
            initial=cell.text()
        note=("eps > 0; momentum: 0–1 или null; affine: true/false; track_running_stats: true (обязательно)."
              if kind=="BatchNorm" else "eps > 0; elementwise_affine и bias: true/false.")
        value,ok=QInputDialog.getMultiLineText(self,kind,note,initial)
        if ok:
            try:
                options=json.loads(value)
                if not isinstance(options,dict) or "type" in options:
                    raise ValueError("Параметры нормализации должны быть объектом JSON без поля type.")
                config=validate_config({"layer_specs":[{"type":"Dense","units":2},
                                                       {"type":kind,**options}],"batch_size":max(2,self.batch_size.value())},self.seed)
                spec=config["layer_specs"][1]
                value=json.dumps({key:v for key,v in spec.items() if key!="type"},ensure_ascii=False)
                cell.setText(value); cell.setToolTip(value)
            except Exception as error:
                QMessageBox.warning(self,"MLP",str(error))

    def move(self,delta):
        row=self.layers.currentRow(); other=row+delta
        if row<0 or other<0 or other>=self.layers.rowCount(): return
        for col in range(self.layers.columnCount()):
            a=self.layers.takeItem(row,col); b=self.layers.takeItem(other,col)
            self.layers.setItem(row,col,b); self.layers.setItem(other,col,a)
        self.layers.setCurrentCell(other,0)

    def remove(self):
        if self.layers.currentRow()>=0: self.layers.removeRow(self.layers.currentRow())

    def load(self,params):
        self.layers.setRowCount(0)
        for spec in params["layer_specs"]: self.append_layer(spec)
        for key in ("output_l1","output_l2","learning_rate","gradient_clip"):
            getattr(self,key).setText(str(params[key]))
        for key in ("epochs","batch_size","random_state"): getattr(self,key).setValue(params[key])
        self.shuffle.setChecked(params["shuffle"])
        self.optimizer.setCurrentText(params["optimizer_name"]); self.loss.setCurrentText(params["loss_name"])
        self.optimizer_params.setPlainText(json.dumps(params["optimizer_params"],indent=2))
        self.loss_params.setPlainText(json.dumps(params["loss_params"],indent=2))
        for key,box in self.train_checks.items(): box.setChecked(key in params["training_metrics"])
        for key,box in self.eval_checks.items(): box.setChecked(not box.isEnabled() or key in params["evaluation_metrics"])
        self.editor.setPlainText(json.dumps(self.collect(),indent=2,ensure_ascii=False))

    def collect(self):
        layers=[]
        for row in range(self.layers.rowCount()):
            values=[self.layers.item(row,c).text() for c in range(7)]
            kind=values[0]
            if kind=="Dense": layers.append(dict(type=kind,units=int(values[1]),activation=values[2],l1=float(values[4]),l2=float(values[5])))
            elif kind=="Activation": layers.append(dict(type=kind,activation=values[2]))
            elif kind=="Dropout": layers.append(dict(type=kind,rate=float(values[3])))
            elif kind in NORMALIZATION:
                options=json.loads(values[6])
                if not isinstance(options,dict) or "type" in options:
                    raise ValueError("Параметры нормализации должны быть объектом JSON без поля type.")
                layers.append(dict(type=kind,**options))
            else: raise ValueError(f"Неизвестный слой: {kind}")
        params=dict(layer_specs=layers,optimizer_name=self.optimizer.currentText(),loss_name=self.loss.currentText(),
                    optimizer_params=json.loads(self.optimizer_params.toPlainText()),loss_params=json.loads(self.loss_params.toPlainText()),
                    shuffle=self.shuffle.isChecked(),training_metrics=[k for k,b in self.train_checks.items() if b.isChecked()],
                    evaluation_metrics=[k for k,b in self.eval_checks.items() if b.isChecked()])
        for key in ("output_l1","output_l2","learning_rate","gradient_clip"): params[key]=float(getattr(self,key).text())
        for key in ("epochs","batch_size","random_state"): params[key]=getattr(self,key).value()
        return validate_config(params,self.seed)

    def sync_tab(self,index):
        previous=self._previous_tab
        try:
            if previous==3 and index!=3:
                self.load(validate_config(json.loads(self.editor.toPlainText()),self.seed))
            elif previous!=3 and index==3:
                self.editor.setPlainText(json.dumps(self.collect(),indent=2,ensure_ascii=False))
        except Exception as error:
            QMessageBox.warning(self,"MLP",str(error))
            with QSignalBlocker(self.tabs): self.tabs.setCurrentIndex(previous)
            return
        self._previous_tab=index

    def checked(self):
        if self.tabs.currentIndex()==3:
            return validate_config(json.loads(self.editor.toPlainText()),self.seed)
        return self.collect()

    def to_json(self):
        try: self.editor.setPlainText(json.dumps(self.collect(),indent=2,ensure_ascii=False))
        except Exception as error: QMessageBox.warning(self,"MLP",str(error))

    def from_json(self):
        try: self.load(validate_config(json.loads(self.editor.toPlainText()),self.seed))
        except Exception as error: QMessageBox.warning(self,"MLP",str(error))

    def apply(self):
        try: self.parameters=self.checked()
        except Exception as error: QMessageBox.warning(self,"MLP",str(error)); return
        self.accept()

    def export(self):
        try:
            params=self.checked()
            path,_=QFileDialog.getSaveFileName(self,"Экспорт MLP","mlp-parameters.json","JSON (*.json)")
            if path: Path(path).write_text(json.dumps(params,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
        except Exception as error: QMessageBox.warning(self,"MLP",str(error))

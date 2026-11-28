"""Model-specific search editor. JSON is a portable representation, not the main UI."""
import json
from copy import deepcopy
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,QTableWidgetItem,
    QHeaderView,QPushButton,QFileDialog,QMessageBox,QDialogButtonBox,QLineEdit)
from .search_space import resolved_fixed,search_rows,validate_space,set_path,unpack_space
from .models import make_model
from .ml_settings_ui import HELP

class SearchDialog(QDialog):
    def __init__(self,name,seed,device,gpu_id,parameters,space,parent=None):
        super().__init__(parent)
        self.name,self.seed,self.device,self.gpu_id=name,seed,device,gpu_id
        self.fixed=resolved_fixed(name,parameters,seed,device,gpu_id)
        self.original=deepcopy(parameters)
        self.rows,self.domains=search_rows(name,self.fixed,device)
        self.parameters=None; self.space=None
        self.setWindowTitle("Optuna · "+name)
        self.resize(1150,760)
        layout=QVBoxLayout(self)
        note=QLabel("Отметьте параметры для подбора. Остальные используют фиксированное значение. Диапазон: список JSON или объект type/low/high/step/log. Сложные и зависимые параметры меняются в редакторе модели.")
        note.setWordWrap(True); layout.addWidget(note)
        self.filter=QLineEdit(); self.filter.setPlaceholderText("Поиск параметра…"); layout.addWidget(self.filter)
        self.table=QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels(["Поиск","Параметр","Фиксированное значение","Диапазон / варианты"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(40)
        header=self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        for i,w in enumerate((70,270,250,480)): self.table.setColumnWidth(i,w)
        header.setStretchLastSection(True)
        self.table.setWordWrap(False)
        layout.addWidget(self.table,1)
        self.filter.textChanged.connect(self.filter_rows)
        row=QHBoxLayout()
        for label,fn in (("Включить все",lambda:self.select(True)),("Выключить все",lambda:self.select(False)),
                         ("Default диапазоны",self.restore),("Сохранить диапазоны JSON…",self.export)):
            b=QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        layout.addLayout(row)
        self.summary=QLabel(); layout.addWidget(self.summary)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Зафиксировать подбор")
        buttons.accepted.connect(self.apply); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.populate(space if space is not None else self.domains)
        self.table.itemChanged.connect(self.update_summary)

    def populate(self,space):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for key,value in self.rows.items():
            r=self.table.rowCount(); self.table.insertRow(r)
            enabled=key in self.domains
            check=QTableWidgetItem()
            check.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|(Qt.ItemIsUserCheckable if enabled else Qt.NoItemFlags))
            check.setCheckState(Qt.Checked if key in space else Qt.Unchecked)
            self.table.setItem(r,0,check)
            for col,val in ((1,key),(2,json.dumps(value,ensure_ascii=False)),
                            (3,json.dumps(space.get(key,self.domains.get(key)),ensure_ascii=False) if enabled else "Фиксированный")):
                item=QTableWidgetItem(val)
                item.setToolTip(HELP.get(key.split(".")[-1],val)+"\n"+val)
                # Nested parents are edited in the model constructor, not alongside their children.
                editable=(col==2 and not isinstance(value,(list,dict)) and key in self.domains) or (col==3 and enabled)
                if not editable: item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r,col,item)
        self.table.blockSignals(False)
        self.update_summary()
        self.filter_rows(self.filter.text())

    def filter_rows(self,text):
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r,text.casefold() not in self.table.item(r,1).text().casefold())

    def update_summary(self,*args):
        count=sum(self.table.item(r,0).checkState()==Qt.Checked for r in range(self.table.rowCount()))
        self.summary.setText(f"В поиске: {count} / {len(self.domains)} доступных параметров")

    def select(self,on):
        for r in range(self.table.rowCount()):
            if self.table.item(r,1).text() in self.domains:
                self.table.item(r,0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def restore(self):
        self.populate(self.domains)

    def collect(self):
        fixed=deepcopy(self.fixed)
        space={}
        for r in range(self.table.rowCount()):
            key=self.table.item(r,1).text()
            if key not in self.domains: continue
            try:
                value=json.loads(self.table.item(r,2).text())
                # None is a library default for parameters not explicitly configured.
                if value is not None or key in fixed or "." in key:
                    set_path(fixed,key,value)
                if self.table.item(r,0).checkState()==Qt.Checked:
                    space[key]=json.loads(self.table.item(r,3).text())
            except (ValueError,TypeError,KeyError,IndexError) as error:
                raise ValueError(f"{key}: некорректное значение JSON: {error}") from error
        space=validate_space(space,self.name,fixed,self.device)
        make_model(self.name,fixed,self.seed,self.device,self.gpu_id)
        # Keep only explicit parameters and changed fixed values, except nested MLP config.
        if self.name!="MLP (pytorch)":
            baseline=resolved_fixed(self.name,self.original,self.seed,self.device,self.gpu_id)
            fixed={k:v for k,v in fixed.items() if k in self.original or k not in baseline or v!=baseline[k]}
        return fixed,space

    def apply(self):
        try: self.parameters,self.space=self.collect()
        except Exception as error:
            QMessageBox.warning(self,"Optuna",str(error)); return
        self.accept()

    def export(self):
        try:
            fixed,space=self.collect()
            path,_=QFileDialog.getSaveFileName(self,"Сохранить диапазоны","optuna-space.json","JSON (*.json)")
            if path:
                Path(path).write_text(json.dumps(dict(format="geogarmodel.search-space.v1",model=self.name,
                    device=self.device,parameters=fixed,space=space),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        except Exception as error: QMessageBox.warning(self,"Optuna",str(error))

"""Editable JSON model defaults with parameter explanations and portable export."""
import json
from pathlib import Path
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPlainTextEdit,
    QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,QDialogButtonBox,
    QPushButton,QFileDialog,QMessageBox,QSplitter)
from PyQt5.QtCore import Qt
from .models import model_defaults, make_model
from .parameter_files import validate_parameters

HELP = {
    "alpha":"Коэффициент регуляризации.",
    "fit_intercept":"Оценивать свободный член.",
    "n_estimators":"Количество деревьев / итераций ансамбля.",
    "iterations":"Количество итераций бустинга.",
    "max_iter":"Максимальное число итераций обучения.",
    "max_depth":"Максимальная глубина дерева; null — ограничение не задано.",
    "depth":"Глубина дерева CatBoost.",
    "learning_rate":"Шаг обучения.",
    "learning_rate_init":"Начальный шаг обучения MLP.",
    "min_samples_leaf":"Минимальное число наблюдений в листе.",
    "min_samples_split":"Минимум наблюдений для разделения узла.",
    "num_leaves":"Максимальное число листьев дерева LightGBM.",
    "max_leaf_nodes":"Максимальное число листовых узлов.",
    "max_features":"Доля или число признаков для поиска разбиения.",
    "subsample":"Доля наблюдений для очередного дерева.",
    "colsample_bytree":"Доля признаков для дерева.",
    "reg_lambda":"L2-регуляризация весов.",
    "reg_alpha":"L1-регуляризация весов.",
    "l2_leaf_reg":"L2-регуляризация листьев CatBoost.",
    "l2_regularization":"L2-регуляризация.",
    "random_state":"Seed случайных операций модели.",
    "random_seed":"Seed случайных операций CatBoost.",
    "n_jobs":"Число CPU-потоков (-1 — все).",
    "thread_count":"Число потоков CatBoost.",
    "hidden_layer_sizes":"Размеры скрытых слоёв sklearn MLP.",
    "activation":"Функция активации.",
    "solver":"Алгоритм оптимизации.",
    "batch_size":"Размер пакета наблюдений.",
    "epochs":"Количество эпох обучения PyTorch.",
    "width":"Ширина скрытого представления нейросети.",
    "layers":"Число скрытых слоёв / блоков.",
    "heads":"Число голов внимания Transformer.",
    "dropout":"Доля отключаемых нейронов при обучении.",
    "weight_decay":"Регуляризация AdamW.",
    "early_stopping":"Досрочная остановка обучения.",
    "tol":"Допуск критерия сходимости.",
    "verbose":"Подробность вывода библиотеки.",
    "verbosity":"Уровень вывода библиотеки.",
    "loss":"Функция потерь.",
    "objective":"Целевая функция.",
    "criterion":"Критерий выбора разбиения.",
    "tree_method":"Метод построения деревьев; GeoGarModel использует hist для XGBoost.",
    "force_col_wise":"Принудительная поколоночная гистограмма LightGBM; GPU использует собственную настройку.",
    "allow_writing_files":"Разрешать CatBoost создавать служебные файлы.",
    "bootstrap":"Выборка с возвращением.",
    "warm_start":"Продолжать предыдущее обучение при повторном fit.",
    "shuffle":"Перемешивать данные при обучении.",
    "validation_fraction":"Доля внутренней проверки для ранней остановки модели.",
    "n_iter_no_change":"Число итераций без улучшения до остановки.",
    "momentum":"Импульс градиентного спуска.",
    "power_t":"Показатель изменения скорости обучения SGD.",
    "epsilon":"Малая константа численной устойчивости.",
    "beta_1":"Первый коэффициент Adam.",
    "beta_2":"Второй коэффициент Adam.",
    "max_bins":"Максимальное число интервалов гистограммы.",
    "min_child_weight":"Минимальный вес дочернего узла XGBoost.",
    "min_child_samples":"Минимум наблюдений в листе LightGBM.",
    "gamma":"Минимальное уменьшение потерь при разбиении XGBoost.",
    "boosting_type":"Тип бустинга LightGBM.",
    "max_delta_step":"Ограничение изменения веса листа XGBoost.",
}


class ParameterDialog(QDialog):
    def __init__(self, name, seed, device, gpu_id, overrides, parent=None):
        super().__init__(parent)
        self.name, self.seed, self.device, self.gpu_id = name, seed, device, gpu_id
        self.defaults = model_defaults(name, seed, device, gpu_id)
        self.parameters = None
        self.setWindowTitle(f"Гиперпараметры · {name} · {device.upper()}")
        self.resize(1040, 720)
        layout = QVBoxLayout(self)
        hint = QLabel("Default — настройки конструктора GeoGarModel и значения, возвращённые библиотекой. "
                      "Редактируйте JSON слева; null означает автоматическое значение, если поддерживается моделью. "
                      "Устройство задаётся отдельно. Optuna заменяет параметры из пространства поиска.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        splitter = QSplitter(Qt.Horizontal)
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(json.dumps({**self.defaults, **overrides}, ensure_ascii=False, indent=2))
        splitter.addWidget(self.editor)
        self.help = QTableWidget(0, 3)
        self.help.setHorizontalHeaderLabels(["Параметр", "Default", "Назначение"])
        self.help.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.help.setAlternatingRowColors(True)
        self.help.setWordWrap(True)
        self.help.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.help.setColumnWidth(0, 145)
        self.help.setColumnWidth(1, 100)
        self.help.setRowCount(len(self.defaults))
        for row, (key, value) in enumerate(sorted(self.defaults.items())):
            for col, item in enumerate((key, json.dumps(value, ensure_ascii=False),
                    HELP.get(key, "Дополнительный параметр библиотеки; см. документацию класса модели."))):
                self.help.setItem(row, col, QTableWidgetItem(item))
        self.help.resizeRowsToContents()
        splitter.addWidget(self.help)
        splitter.setSizes([410, 610])
        layout.addWidget(splitter, 1)
        row = QHBoxLayout()
        reset = QPushButton("Восстановить default")
        reset.clicked.connect(lambda: self.editor.setPlainText(json.dumps(self.defaults, ensure_ascii=False, indent=2)))
        row.addWidget(reset)
        export = QPushButton("Экспорт JSON…")
        export.clicked.connect(self.export_json)
        row.addWidget(export)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Зафиксировать для расчёта")
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def checked(self):
        params = validate_parameters(json.loads(self.editor.toPlainText()))
        json.dumps(params, allow_nan=False)
        make_model(self.name, params, self.seed, self.device, self.gpu_id)
        return params

    def apply(self):
        try:
            self.parameters = self.checked()
        except Exception as error:
            QMessageBox.warning(self, "Гиперпараметры", str(error))
            return
        self.accept()

    def export_json(self):
        try:
            params = self.checked()
            path, _ = QFileDialog.getSaveFileName(self, "Экспорт параметров", self.name+"-parameters.json", "JSON (*.json)")
            if path:
                Path(path).write_text(json.dumps(params, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        except Exception as error:
            QMessageBox.warning(self, "Гиперпараметры", str(error))

"""Context help for the observation, model and solver controls."""
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QMessageBox, QSpinBox, QToolButton


HELP = {
    'mode': 'Выберите физическое свойство, которое требуется восстановить по наблюдённому полю.',
    'method': 'Призмы поддерживают произвольные точки наблюдения. Послойная FFT ускоряет расчёт на регулярной карте с единой высотой.',
    'source': 'Константа создаёт новый куб. Загруженная модель задаёт начальные свойства, геометрию и активные ячейки.',
    'east': 'Столбец восточной координаты точки наблюдения, м.',
    'north': 'Столбец северной координаты точки наблюдения, м.',
    'value': 'Наблюдённая аномалия в мГал для плотности или нТл для магнитного поля.',
    'height': 'Высота датчика Up, м, относительно общего нуля модели. Для FFT все точки лежат на одной высоте.',
    'height_constant': 'Общая высота датчиков Up в метрах, если выбран ввод константы.',
    'error': 'Столбец положительных стандартных ошибок σ в единицах наблюдённого поля или единая константа.',
    'sigma': 'Стандартная ошибка наблюдения в единицах поля. Задаёт вес невязки и масштаб RMS/σ.',
    'crs': 'Метрическая система координат модели и точек. Можно выбрать из списка или ввести код проекции.',
    'stride': 'В расчёт включается каждая N-я строка таблицы. При N = 1 используются все точки.',
    'fft_grid': 'Горизонтальные размеры новой модели задаются по шагу карты или вручную, включая боковой запас.',
    'nx': 'Число ячеек по East в новой сетке или сетке переноса.',
    'ny': 'Число ячеек по North в новой сетке или сетке переноса.',
    'nz': 'Число слоёв между верхом и низом новой модели.',
    'top': 'Глубина верхней грани Down, м, относительно общего нуля модели.',
    'bottom': 'Глубина нижней грани Down, м. Должна быть больше глубины верхней грани.',
    'padding': 'Доля ширины карты, добавляемая с каждой стороны. Значение 0,2 соответствует 20%.',
    'extend_fft': 'Дополняет загруженный куб до заданного бокового запаса с сохранением внутренних свойств.',
    'lower': 'Нижняя граница свойства каждой активной ячейки. Для восприимчивости минимум ≥ 0. Должна быть меньше максимума.',
    'upper': 'Верхняя граница свойства каждой активной ячейки в выбранных единицах модели.',
    'initial': 'Стартовое значение свойства в постоянной модели или добавленных ячейках. Должно лежать в заданных границах.',
    'field': 'Модуль главного магнитного поля, нТл.',
    'inc': 'Наклонение главного поля, градусы. Положительное направление вниз.',
    'dec': 'Склонение главного поля от севера сетки к востоку, градусы.',
    'iterations': 'Предельное число принятых шагов. Начальная оценка имеет номер 0.',
    'depth_index': 'Управляет относительным шагом обновления глубоких ячеек. При 0 все глубины имеют одинаковый множитель.',
    'damping': 'Сила притяжения свойства к опорной модели или нулю. 0 отключает этот штраф. В компактном режиме меняется форма штрафа.',
    'smooth': 'Штраф за разность свойств соседних активных ячеек. При 0 пространственное сглаживание отключено.',
    'vertical_factor': 'Умножает гладкость вдоль Z. При 1 используются общие настройки, при 0 соседи по вертикали свободны.',
    'target': 'Остановка при достижении заданной среднеквадратичной невязки, нормированной на σ. Ориентир 1 при достоверной оценке ошибок.',
    'step': 'Масштаб первой попытки обновления. Решатель уменьшает шаг при недостаточном снижении целевой функции.',
    'offset': 'Подбирает одно постоянное смещение между рассчитанным и наблюдённым полем с учётом σ.',
    'early_stop': 'При слабом относительном улучшении RMS/σ несколько итераций подряд возвращает лучшую завершённую модель.',
    'early_tolerance': 'Минимальное относительное улучшение RMS/σ за итерацию в процентах. Ниже порога растёт счётчик ранней остановки.',
    'early_patience': 'Сколько последовательных слабых улучшений разрешить до остановки. Достаточное улучшение сбрасывает счётчик.',
    'reference_source': 'Опорное свойство определяет, от какой модели измерять отклонения. Для файла требуется та же сетка, СК и покрытие активных ячеек.',
    'penalty_weight': 'Множитель штрафа отклонения от опоры. По глубине он убывает с расстоянием до плоскости наблюдений. По чувствительности используется норма столбца оператора, взвешенного по ошибкам.',
    'weight_power': 'Показатель степени нормированного веса в штрафе малости. При 0 веса одинаковы.',
    'penalty_norm': 'L2 даёт квадратичные штрафы. Компактный режим использует сглаженную L1 для отклонений от опоры. TV использует сглаженную L1 для разностей соседей по осям.',
    'penalty_epsilon': 'Разность в единицах свойства модели, задающая переход от квадратичного поведения к линейному в компактном режиме или TV.',
    'structural': 'Разрешает скачок свойства на гранях, где контраст в опорной модели достигает порога. Сглаживание через эти грани отключается.',
    'freeze_reference': 'Свойства активных ячеек отдельного опорного вокселя фиксируются на всех итерациях. В остальных ячейках опорой служит начальное свойство.',
    'boundary_jump': 'Минимальный абсолютный контраст между соседями в опорной модели для выделения структурной границы, в единицах свойства.',
    'device': 'Выберите процессор или совместимый CUDA GPU для вычислений.',
    'gpu': 'Номер устройства CUDA, начиная с 0. Допустимы номера установленных видеокарт.',
    'memory_policy': 'Автоматический бюджет использует до 50% или 70% свободной памяти с учётом рабочих массивов. В ручном режиме задаётся предел кэша.',
    'cache': 'Максимальный объём кэша в МиБ. 0 отключает кэш, доступный предел зависит от свободной памяти устройства.',
}


class ParameterHelp(QToolButton):
    def __init__(self, anchor, field, title, description):
        super().__init__(anchor)
        self.field = field
        self.title = title
        self.description = description
        self.setText('?')
        self.setFixedSize(20,20)
        self.setStyleSheet('QToolButton {padding: 0; border: 1px solid #7495ae; border-radius: 9px;}')
        self.setAccessibleName('Справка · '+title)
        self.setToolTip(description)
        self.setAccessibleDescription(description)
        self.setProperty('workflowNavigation',True)
        self.clicked.connect(self.explain)
        anchor.installEventFilter(self)
        if isinstance(anchor,QLabel):
            anchor.setContentsMargins(25,0,0,0)
            anchor.setMinimumHeight(22)
        else:
            anchor.setStyleSheet(anchor.styleSheet() + (' QComboBox { padding-right: 48px; }' if isinstance(anchor,QComboBox)
                                                         else ' QCheckBox { padding-right: 25px; }'))

    def eventFilter(self,anchor,event):
        if event.type() in (QEvent.Resize,QEvent.Show):
            x = 0 if isinstance(anchor,QLabel) else anchor.width()-(45 if isinstance(anchor,QComboBox) else 22)
            self.move(max(0,x),max(0,(anchor.height()-20)//2))
        return False

    def explain(self):
        field = self.field
        if isinstance(field,(QSpinBox,QDoubleSpinBox)):
            limits = f'Диапазон ввода от {field.minimum():g} до {field.maximum():g}.'
        elif isinstance(field,QCheckBox):
            limits = 'Можно включить или выключить.'
        elif field.count() <= 12 and not field.isEditable():
            limits = 'Варианты — ' + ', '.join(field.itemText(i) for i in range(field.count())) if field.count() else 'Варианты появятся после загрузки данных.'
        else:
            limits = 'Доступные значения приведены в списке параметра.'
        QMessageBox.information(self,self.title,self.description+'\n\n'+limits)


def install_parameter_help(tab, panel):
    labels = {}
    for form in panel.findChildren(QFormLayout):
        for i in range(form.rowCount()):
            item = form.itemAt(i,QFormLayout.FieldRole)
            label = form.itemAt(i,QFormLayout.LabelRole)
            if item and label and isinstance(label.widget(),QLabel):
                labels[item.widget()] = label.widget()
    for name,description in HELP.items():
        field = tab.initial_panel.source if name=='source' else getattr(tab,name)
        anchor = labels.get(field,field)
        if isinstance(anchor,(QLabel,QCheckBox)):
            title = anchor.text()
        else:
            title = {'mode':'Физическое свойство','method':'Метод расчёта',
                     'source':'Источник начальной модели'}.get(name,'Параметр расчёта')
        ParameterHelp(anchor,field,title,description)

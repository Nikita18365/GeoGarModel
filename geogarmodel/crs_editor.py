"""CRS selection, ArcGIS PRJ import and projection-specific parameters."""
import math
from pathlib import Path
from pyproj import CRS
from pyproj.exceptions import CRSError
from PyQt5.QtWidgets import (QWidget,QFormLayout,QComboBox,QDoubleSpinBox,QPushButton,
    QFileDialog,QLabel,QScrollArea,QMessageBox)
from .coordinates import gsk2011_zones


def read_prj(path):
    """Parse ESRI WKT or OGC WKT without relying on the filename."""
    data=Path(path).read_bytes()
    try: value=data.decode('utf-8-sig')
    except UnicodeDecodeError: value=data.decode('cp1251')
    return CRS.from_wkt(value.strip()).to_wkt()


PARAMETER_LABELS={
    '8801':'Широта начала отсчёта', '8802':'Осевой меридиан',
    '8805':'Масштабный коэффициент', '8806':'Условное восточное смещение',
    '8807':'Условное северное смещение', '8821':'Широта ложного начала',
    '8822':'Долгота ложного начала', '8823':'Первая стандартная параллель',
    '8824':'Вторая стандартная параллель', '8826':'Восточное смещение ложного начала',
    '8827':'Северное смещение ложного начала', '8811':'Широта центра проекции',
    '8812':'Долгота центра проекции', '8813':'Азимут начальной линии',
    '8814':'Угол поворота сетки', '8815':'Масштаб на начальной линии',
    '8816':'Восточное смещение центра', '8817':'Северное смещение центра',
}


def projection_parameters(crs):
    return CRS.from_user_input(crs).to_json_dict().get('conversion',{}).get('parameters',[])


def with_projection_parameters(crs,changes):
    """Change only requested conversion parameters; retain datum and units."""
    crs=CRS.from_user_input(crs); data=crs.to_json_dict()
    parameters=data.get('conversion',{}).get('parameters',[])
    modified=False
    for index,value in changes.items():
        if not math.isfinite(value): raise ValueError('Параметры проекции должны быть конечными числами.')
        if not 0<=index<len(parameters): raise ValueError('Неизвестный параметр проекции.')
        if parameters[index]['value']!=value:
            parameters[index]['value']=value; modified=True
    if not modified: return crs.to_string()
    data.pop('id',None); data['conversion'].pop('id',None)
    data['name']=crs.name.removesuffix(' (пользовательские параметры)')+' (пользовательские параметры)'
    return CRS.from_json_dict(data).to_wkt()


def false_easting(crs):
    crs=CRS.from_user_input(crs)
    if crs.coordinate_operation:
        for parameter in crs.coordinate_operation.params:
            if parameter.code in ('8806','8826'):
                return parameter.value*parameter.unit_conversion_factor
    return None


def with_false_easting(crs,value):
    crs=CRS.from_user_input(crs)
    if not math.isfinite(value): raise ValueError('False East должен быть конечным числом.')
    if not crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in crs.axis_info[:2]):
        raise ValueError('Для False East выберите плоскую СК в метрах.')
    data=crs.to_json_dict()
    for parameter in data.get('conversion',{}).get('parameters',[]):
        if str(parameter.get('id',{}).get('code')) in ('8806','8826'):
            if false_easting(crs)==value: return crs.to_string()
            parameter['value']=value; parameter['unit']='metre'
            data.pop('id',None)
            data['conversion'].pop('id',None)
            data['name']=crs.name.split(' [False East=')[0]+f' [False East={value:g} m]'
            return CRS.from_json_dict(data).to_wkt()
    raise ValueError('В этой проекции нет поддерживаемого параметра False East.')


class ParameterSpinBox(QDoubleSpinBox):
    def textFromValue(self,value):
        text=super().textFromValue(value)
        return text.rstrip('0').rstrip(self.locale().decimalPoint()) if self.locale().decimalPoint() in text else text


class CRSEditor(QWidget):
    def __init__(self,known='',parent=None):
        super().__init__(parent)
        form=QFormLayout(self); form.setContentsMargins(0,0,0,0)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.crs=QComboBox(); self.crs.setEditable(True)
        self.crs.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.crs.setMinimumContentsLength(28)
        for code,name in gsk2011_zones(): self.crs.addItem(code+' | '+name,code)
        self.crs.setCurrentText('')
        form.addRow('Система координат (EPSG / PROJ / WKT)',self.crs)
        self.import_button=QPushButton('Импорт СК из .prj (ArcGIS)…'); self.import_button.clicked.connect(self.import_prj)
        form.addRow(self.import_button)
        self.description=QLabel(); self.description.setWordWrap(True); form.addRow(self.description)
        self.parameter_container=QWidget(); self.parameter_form=QFormLayout(self.parameter_container)
        self.parameter_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.parameter_form.setContentsMargins(0,0,8,0)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.parameter_container)
        scroll.setMinimumHeight(160); scroll.setMaximumHeight(300); form.addRow(scroll)
        self.fields={}; self.defaults={}
        self.crs.currentTextChanged.connect(self.sync)
        if known: self.set_crs(known)
        else: self.sync()

    def definition(self):
        index=self.crs.currentIndex()
        if index>=0 and self.crs.currentText()==self.crs.itemText(index):
            return self.crs.itemData(index)
        return self.crs.currentText().split('|')[0].strip()

    def set_crs(self,value):
        crs=CRS.from_user_input(value)
        text=crs.name
        self.crs.blockSignals(True)
        self.crs.addItem(text,crs.to_wkt()); self.crs.setCurrentIndex(self.crs.count()-1)
        self.crs.blockSignals(False); self.sync()

    def import_prj(self):
        path,_=QFileDialog.getOpenFileName(self,'Импорт системы координат ArcGIS','','Проекция (*.prj);;Все файлы (*)')
        if not path: return
        try: self.set_crs(read_prj(path))
        except (CRSError,ValueError,OSError,UnicodeError) as error:
            QMessageBox.warning(self,'Импорт СК','Не удалось прочитать СК из .prj:\n'+str(error))

    def sync(self):
        while self.parameter_form.rowCount(): self.parameter_form.removeRow(0)
        self.fields={}; self.defaults={}
        try: crs=CRS.from_user_input(self.definition())
        except (CRSError,ValueError,TypeError):
            self.description.setText('Выберите СК, введите EPSG / PROJ / WKT или импортируйте .prj.'); return
        self.description.setText(crs.name+('\nПроекция: '+crs.coordinate_operation.method_name if crs.coordinate_operation else ''))
        for i,parameter in enumerate(projection_parameters(crs)):
            unit=parameter.get('unit',''); unit=unit if isinstance(unit,str) else unit.get('name','')
            unit={'metre':'м','meter':'м','degree':'°','unity':''}.get(unit.lower(),unit)
            label=PARAMETER_LABELS.get(str(parameter.get('id',{}).get('code')),parameter['name'])
            field=ParameterSpinBox(); field.setDecimals(10); field.setRange(-1e12,1e12)
            field.setGroupSeparatorShown(True); field.setKeyboardTracking(False)
            field.setValue(parameter['value']); field.setToolTip(parameter['name'])
            self.parameter_form.addRow(label+(', '+unit if unit else ''),field)
            self.fields[i]=field; self.defaults[i]=field.value()
        if not self.fields: self.parameter_form.addRow(QLabel('У этой СК нет числовых параметров проекции.'))

    def value(self):
        value=self.definition()
        if not value: raise ValueError('Укажите систему координат.')
        return with_projection_parameters(value,{i:w.value() for i,w in self.fields.items() if w.value()!=self.defaults[i]})

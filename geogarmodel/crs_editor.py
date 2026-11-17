"""Shared CRS and false-easting controls for map projects and source layers."""
import math
from pyproj import CRS
from PyQt5.QtWidgets import QWidget,QFormLayout,QComboBox,QCheckBox,QDoubleSpinBox
from .coordinates import gsk2011_zones


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


class CRSEditor(QWidget):
    def __init__(self,known='',parent=None):
        super().__init__(parent)
        form=QFormLayout(self); form.setContentsMargins(0,0,0,0)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.crs=QComboBox(); self.crs.setEditable(True)
        self.crs.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.crs.setMinimumContentsLength(28)
        for code,name in gsk2011_zones(): self.crs.addItem(code+' | '+name)
        self.crs.setCurrentText(known)
        self.crs.lineEdit().setCursorPosition(0)
        self.override=QCheckBox('Задать False East вручную, м')
        self.east=QDoubleSpinBox(); self.east.setDecimals(3); self.east.setRange(-1e12,1e12)
        self.east.setGroupSeparatorShown(True)
        self.east.setToolTip('Полное значение параметра проекции, а не величина сдвига координат.')
        form.addRow('Система координат (EPSG / PROJ / WKT)',self.crs)
        form.addRow(self.override); form.addRow('False East (условное восточное смещение)',self.east)
        self.crs.currentTextChanged.connect(self.sync)
        self.override.toggled.connect(self.east.setEnabled)
        self.sync()

    def sync(self):
        self.override.setChecked(False); self.east.setEnabled(False)
        try: value=false_easting(self.crs.currentText().split('|')[0].strip())
        except Exception: value=None
        self.east.setValue(value if value is not None else 0)

    def value(self):
        value=self.crs.currentText().split('|')[0].strip()
        if not value: raise ValueError('Укажите систему координат.')
        return with_false_easting(value,self.east.value()) if self.override.isChecked() else CRS.from_user_input(value).to_string()

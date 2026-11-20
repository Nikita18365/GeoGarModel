"""CRS selection from PROJ, without guessing unavailable zone codes."""
from functools import lru_cache
from pyproj import CRS
from pyproj.database import query_crs_info
from pyproj.enums import PJType


@lru_cache(maxsize=1)
def gsk2011_zones():
    infos = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])
    zones = [i for i in infos if i.name.startswith("GSK-2011 / Gauss-Kruger zone ")]
    return sorted([(f"EPSG:{i.code}", i.name) for i in zones], key=lambda i: int(i[1].rsplit(" ", 1)[1]))


def validate_crs(value, coordinate_unit):
    if not value.strip():
        return ""
    crs = CRS.from_user_input(value)
    if coordinate_unit == "m" and (not crs.is_projected or
            any(abs(a.unit_conversion_factor - 1.) > 1e-9 for a in crs.axis_info[:2])):
        raise ValueError("Для координат в метрах выберите метрическую проекцию.")
    if coordinate_unit == "degrees" and not crs.is_geographic:
        raise ValueError("Для градусов выберите географическую CRS.")
    return crs.to_string()

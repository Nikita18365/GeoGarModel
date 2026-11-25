"""Presentation-only settings for publication maps and tight image export."""
from pathlib import Path
import math
from functools import lru_cache
from matplotlib import font_manager
from .model_archive import atomic_output

FONT_ELEMENTS={
    'title':('Название карты',12,'#111111'),
    'colorbar':('Значения цветовой шкалы',10,'#222222'),
    'colorbar_label':('Подпись цветовой шкалы',10,'#222222'),
    'coordinates':('Значения координатной сетки',10,'#222222'),
    'axes':('Подписи осей',10,'#222222'),
    'scale_unit':('Размерность масштабной линейки',9,'#222222'),
    'scale_crs':('СК под масштабной линейкой',8,'#222222'),
    'scale_values':('Значения масштабной линейки',8,'#222222'),
    'contours':('Значения изолиний',8,None),
}


DEFAULT_MAP_FONT = 'DejaVu Sans'


@lru_cache(maxsize=1)
def available_map_fonts():
    """Real family names known to Matplotlib, not Qt/Windows aliases."""
    return tuple(sorted(font_manager.fontManager.get_font_names(), key=str.casefold))


@lru_cache(maxsize=256)
def resolve_map_font(family):
    """Resolve before drawing, rather than repeatedly logging missing fonts."""
    names = {name.casefold(): name for name in available_map_fonts()}
    requested = (family or '').strip().casefold()
    if requested in names:
        return names[requested]
    aliases = {
        'ms shell dlg 2': ('Tahoma', 'Segoe UI'),
        'ms shell dlg': ('Microsoft Sans Serif', 'Tahoma', 'Segoe UI'),
    }
    for candidate in (*aliases.get(requested, ()), DEFAULT_MAP_FONT):
        if candidate.casefold() in names:
            return names[candidate.casefold()]
    # Matplotlib normally ships DejaVu Sans even on a minimal installation.
    return font_manager.FontProperties(fname=font_manager.findfont(
        font_manager.FontProperties(family=['sans-serif']))).get_name()


def migrate_map_fonts(settings):
    """Split the legacy combined colorbar style without changing old projects."""
    result=dict(settings or {})
    if 'colorbar' in result and 'colorbar_label' not in result:
        result['colorbar_label']=dict(result['colorbar'])
    for key,style in result.items():
        if isinstance(style,dict) and 'family' in style:
            result[key]=dict(style,family=resolve_map_font(style['family']))
    return result


def text_options(settings,key,size=None,color=None):
    _,default_size,default_color=FONT_ELEMENTS[key]
    style=settings.get(key,{})
    options=dict(fontfamily=resolve_map_font(style.get('family',DEFAULT_MAP_FONT)),
                 fontsize=style.get('size',default_size if size is None else size),
                 fontweight='bold' if style.get('bold',False) else 'normal',
                 fontstyle='italic' if style.get('italic',False) else 'normal')
    selected=style.get('color') or default_color or color
    if selected: options['color']=selected
    return options


def apply_map_fonts(figure,settings):
    if not figure.axes: return
    ax=figure.axes[0]
    def apply(text,key):
        if key in settings: text.update(text_options(settings,key,color=text.get_color()))
    apply(ax.title,'title')
    for axis in (ax.xaxis,ax.yaxis):
        apply(axis.label,'axes')
        for label in [*axis.get_ticklabels(),axis.get_offset_text()]: apply(label,'coordinates')
        if 'coordinates' in settings:
            options=text_options(settings,'coordinates')
            axis.set_tick_params(labelsize=options['fontsize'],labelcolor=options['color'])
    for label in ax.texts:
        if label.get_gid()=='map-contour-label': apply(label,'contours')
    for other in figure.axes[1:]:
        if not hasattr(other,'_colorbar'): continue
        for axis in (other.xaxis,other.yaxis):
            apply(axis.label,'colorbar_label')
            for label in [*axis.get_ticklabels(),axis.get_offset_text()]: apply(label,'colorbar')
            if 'colorbar' in settings:
                options=text_options(settings,'colorbar')
                axis.set_tick_params(labelsize=options['fontsize'],labelcolor=options['color'])


def centered_limits(view,factor):
    if not math.isfinite(factor) or factor<=0: raise ValueError('Некорректный масштаб.')
    def axis(bounds):
        a,b=bounds; center=a+(b-a)/2
        return (center-(b-a)*factor/2,center+(b-a)*factor/2)
    return tuple(axis(bounds) for bounds in view)


def save_map_image(figure,path,*,dpi=300,padding_mm=0,transparent=False,jpeg_quality=95):
    """Export the live publication figure; no Qt canvas access from a worker."""
    path=Path(path)
    fmt=path.suffix.lower().lstrip('.')
    if fmt not in ('png','jpg','jpeg','tif','tiff','pdf','svg'):
        raise ValueError('Неподдерживаемый формат изображения.')
    if not 72<=dpi<=1200 or not 0<=padding_mm<=30 or not 1<=jpeg_quality<=100:
        raise ValueError('Проверьте DPI, отступ и качество JPEG.')
    options=dict(format=fmt,dpi=dpi,bbox_inches='tight',pad_inches=padding_mm/25.4,
                 transparent=transparent and fmt not in ('jpg','jpeg'))
    if fmt in ('jpg','jpeg'): options['pil_kwargs']={'quality':jpeg_quality}
    # Guard very large rasters before savefig allocates the target canvas.
    if fmt not in ('pdf','svg'):
        width,height=figure.get_size_inches()
        if (width*dpi+2*padding_mm*dpi/25.4)*(height*dpi+2*padding_mm*dpi/25.4)>100_000_000:
            raise ValueError('Изображение превышает 100 млн пикселей. Уменьшите DPI или размер окна.')
    with atomic_output(path) as temporary:
        figure.savefig(temporary,**options)

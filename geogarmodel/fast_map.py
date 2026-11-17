"""Interactive map viewport: persistent Qt scene and downsampled raster display."""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF,QTimer
from PyQt5.QtGui import QPainterPath,QPen,QColor,QBrush,QPainter
from PyQt5.QtWidgets import QGraphicsPathItem
from .plotting import map_colors


class MapImage(pg.ImageItem):
    smooth=False

    def paint(self,painter,*args):
        painter.save(); painter.setRenderHint(QPainter.SmoothPixmapTransform,self.smooth)
        super().paint(painter,*args); painter.restore()


class FastMap(pg.PlotWidget):
    def __init__(self,workbench):
        super().__init__(background='#101c2f'); self.workbench=workbench
        self.setLabel('bottom','X, м'); self.setLabel('left','Y, м')
        self.setAspectLocked(True); self.showGrid(x=True,y=True,alpha=.12)
        self.getPlotItem().setMenuEnabled(False)
        self.getViewBox().sigRangeChangedManually.connect(self.sync_view)
        self.updating=False
        self.rasters=[]; self.refresh_timer=QTimer(self); self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(40); self.refresh_timer.timeout.connect(self.update_rasters)
        self.getViewBox().sigRangeChanged.connect(lambda *args:self.refresh_timer.start())
        self.bar=pg.ColorBarItem(values=(0,1),interactive=False); self.bar.setParentItem(self.getPlotItem())
        self.getPlotItem().layout.addItem(self.bar,2,3)

    def sync_view(self,*args):
        if self.updating: return
        x,y=self.viewRange(); window=self.workbench
        window._map_view=(tuple(x),tuple(y))
        if window.figure.axes:
            window.figure.axes[0].set_xlim(x); window.figure.axes[0].set_ylim(y)
        self.refresh_timer.start()

    def update_rasters(self):
        x,y=self.viewRange()
        for image,grid,norm in self.rasters:
            ix=np.clip(np.searchsorted(grid.x,sorted(x)),0,len(grid.x)); iy=np.clip(np.searchsorted(grid.y,sorted(y)),0,len(grid.y))
            x0=max(0,ix[0]-1); x1=min(len(grid.x),ix[1]+1); y0=max(0,iy[0]-1); y1=min(len(grid.y),iy[1]+1)
            if x1<=x0 or y1<=y0: image.hide(); continue
            image.show(); step=max(1,int(np.ceil(max((x1-x0)/max(self.width(),200),(y1-y0)/max(self.height(),200)))))
            key=(x0,x1,y0,y1,step)
            if getattr(image,'sample_key',None)==key: continue
            data=np.asarray(norm(grid.values[y0:y1:step,x0:x1:step]),dtype=np.float32)
            data=data/getattr(image,'display_divisor',1)
            data[~np.isfinite(grid.values[y0:y1:step,x0:x1:step])]=np.nan
            image.setImage(data,autoLevels=False,levels=(0,1))
            image.setRect(QRectF(grid.x[x0]-grid.dx/2,grid.y[y0]-grid.dy/2,
                min(step*data.shape[1],x1-x0)*grid.dx,min(step*data.shape[0],y1-y0)*grid.dy))
            image.sample_key=key

    def show_layers(self,rows,view=None):
        self.updating=True
        try:
            self.clear(); self.rasters=[]
            self.bar.setVisible(any(r['visible'] and r['kind']=='grid' for r in rows))
            self.setTitle(self.workbench.map_title.text().strip() or self.workbench.map_project_name,color='#dbe7f6')
            for index,row in enumerate(reversed(rows)):
                if not row['visible']: continue
                value=row['data']; style=row['style']
                if row['kind']=='grid':
                    cm,norm=map_colors(value,**style)
                    # Display copy only; calculations retain original values/precision.
                    image=MapImage(axisOrder='row-major'); image.setAutoDownsample(True)
                    from matplotlib.colors import BoundaryNorm
                    image.display_divisor=max(1,cm.N-1) if isinstance(norm,BoundaryNorm) else 1
                    image.setLookupTable((cm(np.linspace(0,1,256))*255).astype(np.uint8))
                    image.setRect(QRectF(value.x[0]-value.dx/2,value.y[0]-value.dy/2,len(value.x)*value.dx,len(value.y)*value.dy))
                    image.setOpacity(style.get('alpha',1)); image.setZValue(index*10)
                    image.smooth=style.get('smooth',False)
                    self.addItem(image)
                    self.rasters.append((image,value,norm))
                    raw=np.linspace(norm.vmin,norm.vmax,256)
                    self.bar.setColorMap(pg.ColorMap(np.linspace(0,1,256),(cm(norm(raw))*255).astype(np.uint8)))
                    self.bar.setLevels((norm.vmin,norm.vmax))
                    if style.get('hillshade'):
                        from matplotlib.colors import LightSource
                        filled=np.where(np.isfinite(value.values),value.values,np.nanmedian(value.values))
                        shade=LightSource(style.get('sun_azimuth',315),style.get('sun_altitude',45)).hillshade(filled,dx=value.dx,dy=-value.dy,vert_exag=style.get('sun_relief',1))
                        shade[~np.isfinite(value.values)]=np.nan
                        shadow=pg.ImageItem(shade.astype(np.float32),axisOrder='row-major',autoLevels=False,levels=(0,1))
                        shadow.setAutoDownsample(True); shadow.setRect(image.boundingRect())
                        shadow.setRect(QRectF(value.x[0]-value.dx/2,value.y[0]-value.dy/2,len(value.x)*value.dx,len(value.y)*value.dy))
                        shadow.setOpacity(style.get('sun_intensity',.25)*style.get('alpha',1)); shadow.setZValue(index*10+1); self.addItem(shadow)
                else:
                    def geometry(g):
                        if not g: return
                        kind=g['type']; coords=g.get('coordinates',[])
                        if kind=='GeometryCollection':
                            for child in g['geometries']: geometry(child)
                        elif kind.startswith('Multi'):
                            for child in coords: geometry(dict(type=kind[5:],coordinates=child))
                        elif kind=='Point':
                            point=pg.ScatterPlotItem([coords[0]],[coords[1]],brush=style.get('line_color','#fbbf24'),size=6)
                            point.setZValue(index*10+2); self.addItem(point)
                        else:
                            path=QPainterPath()
                            for ring in coords if kind=='Polygon' else [coords]:
                                if not ring: continue
                                path.moveTo(*ring[0][:2])
                                for xy in ring[1:]: path.lineTo(*xy[:2])
                                if kind=='Polygon': path.closeSubpath()
                            item=QGraphicsPathItem(path); pen=QPen(QColor(style.get('line_color','#fbbf24')))
                            pen.setWidthF(style.get('line_width',1)); pen.setCosmetic(True); item.setPen(pen)
                            if kind=='Polygon' and style.get('fill'):
                                color=QColor(style.get('fill_color','#fbbf24')); color.setAlphaF(style.get('fill_alpha',.3)); item.setBrush(QBrush(color))
                            item.setZValue(index*10+2); self.addItem(item)
                    for g in value.features: geometry(g)
            if view: self.setRange(xRange=view[0],yRange=view[1],padding=0)
            else: self.autoRange(padding=.02)
            # Reuse computed contour geometry and labels from the export renderer.
            axes=self.workbench.figure.axes
            if axes:
                for collection in axes[0].collections:
                    if not hasattr(collection,'get_paths'): continue
                    for path in collection.get_paths():
                        item=QGraphicsPathItem(); shape=QPainterPath()
                        for vertices,code in path.iter_segments(curves=False):
                            if code==1: shape.moveTo(*vertices[:2])
                            elif code==79: shape.closeSubpath()
                            else: shape.lineTo(*vertices[:2])
                        item.setPath(shape); color=collection.get_edgecolor()
                        if len(color):
                            pen=QPen(QColor.fromRgbF(*color[0])); pen.setCosmetic(True); pen.setWidthF(float(collection.get_linewidth()[0])); item.setPen(pen)
                        item.setZValue(collection.get_zorder()); self.addItem(item)
                for text in axes[0].texts:
                    if text.get_transform()!=axes[0].transData: continue
                    label=pg.TextItem(text.get_text(),color=text.get_color(),anchor=(.5,.5)); label.setPos(*text.get_position()); label.setZValue(1000); self.addItem(label)
        finally: self.updating=False
        self.update_rasters()

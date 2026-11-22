"""Interactive map viewport: persistent Qt scene and downsampled raster display."""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF,QTimer,QPointF,Qt
from PyQt5.QtGui import QPainterPath,QPen,QColor,QBrush,QPainter,QPolygonF,QFont
from PyQt5.QtWidgets import QGraphicsPathItem
from .plotting import map_colors
from .map_sampling import sample_display


class NorthMark(pg.GraphicsObject):
    direction=np.array([0.,-1.])

    def boundingRect(self): return QRectF(-18,-44,56,42)

    def paint(self,painter,*args):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.boundingRect(),QColor('white'))
        direction=self.direction; center=np.array([0.,-23.]); base=center-direction*14.5; tip=center+direction*14.5
        side=np.array([-direction[1],direction[0]])*7
        painter.setPen(QPen(QColor('black'),3)); painter.drawLine(QPointF(*base),QPointF(*(tip-direction*9)))
        painter.setBrush(QBrush(QColor('black'))); painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([QPointF(*tip),QPointF(*(tip-direction*13+side)),QPointF(*(tip-direction*13-side))]))
        painter.setPen(QColor('black')); font=QFont(); font.setPixelSize(20); font.setBold(True); painter.setFont(font)
        painter.drawText(QPointF(max(base[0],tip[0])+8,-16),'N')


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
        self.north=NorthMark(); self.north.setParentItem(self.bar); self.north.setZValue(2000)
        self.getPlotItem().layout.setRowMinimumHeight(0,48)
        self.geo_items=[]

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
            ratio=self.devicePixelRatioF(); width=max(2,int(self.getViewBox().width()*ratio)); height=max(2,int(self.getViewBox().height()*ratio))
            # Two samples per source cell suffice when zooming; QPainter handles
            # the final subpixel stretch without repeatedly expanding tiny ROIs.
            if image.smooth:
                width=min(width,max(2,int(np.ceil((max(x)-min(x))/grid.dx))*2))
                height=min(height,max(2,int(np.ceil((max(y)-min(y))/grid.dy))*2))
            key=(*x,*y,width,height,image.smooth)
            if getattr(image,'sample_key',None)==key: continue
            sampled=sample_display(grid,(*sorted(x),*sorted(y)),width,height,image.smooth)
            if sampled is None: image.hide(); continue
            image.show(); values,extent=sampled
            data=np.asarray(norm(values),dtype=np.float32)
            data=data/getattr(image,'display_divisor',1)
            data[~np.isfinite(values)]=np.nan
            image.setImage(data,autoLevels=False,levels=(0,1))
            xmin,xmax,ymin,ymax=extent
            image.setRect(QRectF(xmin,ymin,xmax-xmin,ymax-ymin))
            image.sample_key=key
        if self.workbench.refresh_map_geography(strict=False):
            self.workbench.canvas.draw_idle()
        self.update_north()

    def update_north(self):
        from .map_navigation import north_direction
        try:
            x,y=self.viewRange(); cx,cy=np.mean(x),np.mean(y)
            direction=north_direction(self.workbench.map_project_crs,cx,cy)
            a=self.getViewBox().mapViewToScene(QPointF(cx,cy)); b=self.getViewBox().mapViewToScene(QPointF(cx+direction[0],cy+direction[1]))
            d=np.array([b.x()-a.x(),b.y()-a.y()]); self.north.direction=d/np.linalg.norm(d)
            self.north.setPos(self.bar.vb.mapToItem(self.bar,QPointF(self.bar.vb.width()*.25,0)))
            self.north.setVisible(bool(self.rasters)); self.north.update()
        except (ValueError,RuntimeError): self.north.hide()

    def draw_geography(self,data,settings):
        from .graticule import degree_label
        for item in self.geo_items: self.removeItem(item)
        self.geo_items=[]
        for name,key in [('bottom','xticks'),('left','yticks')]:
            axis=self.getAxis(name)
            if settings['axes']=='geographic': axis.setTicks([[(p,degree_label(v,settings['notation'])) for p,v in data[key]]] if data else [])
            else: axis.setTicks(None)
        self.setLabel('bottom','Долгота' if settings['axes']=='geographic' else 'X, м')
        self.setLabel('left','Широта' if settings['axes']=='geographic' else 'Y, м')
        if not data: return
        mode=settings['mode']; pen=pg.mkPen(settings['color'],width=max(.1,settings['width_microns']*96/25400),style=Qt.DotLine if mode=='dotted' else Qt.SolidLine)
        if mode in ('solid','dotted'):
            for x,y in data['curves']:
                item=pg.PlotCurveItem(x,y,pen=pen,connect='finite'); item.setZValue(1500); self.addItem(item); self.geo_items.append(item)
        elif mode in ('crosses','edges'):
            groups=[(data['crosses'],'+')]
            if mode=='edges':
                edges=np.asarray(data['edges']); groups=[]
                if len(edges):
                    lo,hi=self.workbench.figure.axes[0].get_ylim()
                    vertical=np.isclose(edges[:,1],lo)|np.isclose(edges[:,1],hi)
                    groups=[(edges[vertical].T,'|'),(edges[~vertical].T,'_')]
            for xy,symbol in groups:
                if len(xy)!=2: continue
                item=pg.ScatterPlotItem(xy[0],xy[1],symbol=symbol,pen=pen,brush=None,size=8)
                item.setZValue(1500); self.addItem(item); self.geo_items.append(item)

    def show_layers(self,rows,view=None):
        self.updating=True
        try:
            self.clear(); self.rasters=[]; self.geo_items=[]
            self.bar.setVisible(any(r['visible'] and r['kind']=='grid' for r in rows))
            self.setTitle(self.workbench.map_title.text().strip() or self.workbench.map_project_name,color='#dbe7f6')
            self.getPlotItem().layout.setRowFixedHeight(0,58)
            for index,row in enumerate(reversed(rows)):
                if not row['visible']: continue
                value=row['data']; style=row['style']
                if row['kind']=='grid':
                    cm,norm=map_colors(value,**style)
                    # Display copy only; calculations retain original values/precision.
                    image=MapImage(axisOrder='row-major'); image.setAutoDownsample(False)
                    from matplotlib.colors import BoundaryNorm
                    image.display_divisor=max(1,cm.N-1) if isinstance(norm,BoundaryNorm) else 1
                    image.setLookupTable((cm(np.linspace(0,1,256))*255).astype(np.uint8))
                    image.setRect(QRectF(value.x[0]-value.dx/2,value.y[0]-value.dy/2,len(value.x)*value.dx,len(value.y)*value.dy))
                    image.setOpacity(style.get('alpha',1)); image.setZValue(index*10)
                    image.smooth=style.get('smooth',True)
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
                        shadow=MapImage(shade.astype(np.float32),axisOrder='row-major',autoLevels=False,levels=(0,1)); shadow.smooth=image.smooth
                        shadow.setAutoDownsample(True); shadow.setRect(image.boundingRect())
                        shadow.setRect(QRectF(value.x[0]-value.dx/2,value.y[0]-value.dy/2,len(value.x)*value.dx,len(value.y)*value.dy))
                        shadow.setOpacity(style.get('sun_intensity',.25)*style.get('alpha',1)); shadow.setZValue(index*10+1); self.addItem(shadow)
                elif row['kind']=='raster':
                    west,south,east,north=value.bounds
                    image=MapImage(np.ascontiguousarray(value.image[::-1]),axisOrder='row-major',
                                   autoLevels=False,levels=None)
                    image.setRect(QRectF(west,south,east-west,north-south))
                    image.setZValue(index*10)
                    self.addItem(image)
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
                    if collection in getattr(axes[0],'_geo_artists',[]): continue
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
        self.draw_geography(getattr(self.workbench,'_map_geo_data',None),self.workbench.read_geography())
        self.update_rasters()

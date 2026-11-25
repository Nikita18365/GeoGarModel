"""Frame-relative publication layout and manual decoration positions.

Offsets are physical inches, so dragging does not change field coordinates,
scale length, map limits or export DPI. The colourbar locator is evaluated
after the map's equal aspect has been applied.
"""
import re
import numpy as np
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.backend_bases import MouseButton


MOVABLE=('title','xlabel','ylabel','scale','north','colorbar_vertical','colorbar_horizontal')


def clean_crs_name(name):
    return re.sub(r'\s*\(пользовательские параметры\)', '', name, flags=re.IGNORECASE).strip()


def validated_offsets(value):
    result={}
    if not isinstance(value,dict): return result
    for key,offset in value.items():
        if key not in MOVABLE: continue
        try:
            xy=np.asarray(offset,dtype=float)
            if xy.shape==(2,) and np.isfinite(xy).all() and np.max(np.abs(xy))<=100:
                result[key]=xy.tolist()
        except (TypeError,ValueError): pass
    return result


def offset_pixels(window,key):
    return np.asarray(window.map_offsets.get(key,(0.,0.)))*window.figure.dpi


def artist_bbox(artists,renderer):
    boxes=[]
    for artist in artists:
        if not artist.get_visible(): continue
        box=artist.get_window_extent(renderer)
        if np.isfinite(box.extents).all() and (box.width>0 or box.height>0):
            boxes.append(box)
    return Bbox.union(boxes) if boxes else None


def attach_colorbar(window,ax,image,unit):
    figure=ax.figure
    orientation=window.colorbar_orientation.currentData()
    cax=figure.add_axes([.8,.2,.02,.5],label='map-colorbar')
    figure.colorbar(image,cax=cax,orientation=orientation,label=unit)
    cax.set_box_aspect(None)
    cax.set_aspect('auto')
    cax.set_navigate(False)

    def locate(_axes,renderer):
        if renderer is None:
            renderer=window.canvas.get_renderer()
        ax.apply_aspect()
        frame=ax.get_window_extent(renderer)
        p=renderer.points_to_pixels(1)
        if orientation=='vertical':
            # Fixed typographic gap to the *actual* right edge, not subplot slot.
            box=Bbox.from_bounds(frame.x1+10*p,frame.y0,11*p,frame.height)
        else:
            ax.xaxis.get_tightbbox(renderer)
            update=getattr(ax,'_map_scale_update',None)
            if update: update(renderer)
            scale=artist_bbox(getattr(ax,'_map_scale_artists',[]),renderer)
            # Reserve enough space for the end tick labels inside the map edge.
            tick_width=max((t.get_window_extent(renderer).width for t in cax.get_xticklabels()
                            if t.get_visible()),default=0.)
            right=frame.x1-max(5*p,tick_width/2)
            left=max(frame.x0+frame.width*.54,frame.x0+5*p)
            bottom=ax.xaxis.label.get_window_extent(renderer).y0-20*p
            if scale is not None:
                left=max(left,scale.x1+max(12*p,tick_width/2+6*p))
                patches=getattr(ax,'_map_scale_artists',[])
                if patches: bottom=patches[0].get_window_extent(renderer).y0
            if right-left<max(frame.width*.19,24*p):
                # Extremely long CRS, large fonts, or a deliberately long
                # scale bar: use a second row rather than obscure either.
                left=frame.x0+frame.width*.45
                right=max(left+10*p,right)
                if scale is not None: bottom=scale.y0-16*p
            box=Bbox.from_bounds(left,bottom,max(right-left,1.),8*p)
        dx,dy=offset_pixels(window,'colorbar_'+orientation)
        box=box.translated(dx,dy)
        return box.transformed(figure.transFigure.inverted())

    cax.set_axes_locator(locate)
    return cax


def apply_text_offsets(window):
    if not window.figure.axes: return
    ax=window.figure.axes[0]
    for key,text in [('title',ax.title),('xlabel',ax.xaxis.label),('ylabel',ax.yaxis.label)]:
        if not hasattr(text,'_map_base_transform'):
            text._map_base_transform=text.get_transform()
        dx,dy=window.map_offsets.get(key,(0.,0.))
        text.set_transform(text._map_base_transform+ScaledTranslation(dx,dy,window.figure.dpi_scale_trans))


def fit_publication(figure,renderer):
    """Only reduce the subplot slot if displayed decorations leave the canvas.
    Never invoke tight_layout on the independent colorbar axes.
    """
    content=figure.get_tightbbox(renderer).transformed(figure.dpi_scale_trans)
    frame=figure.bbox
    if not np.isfinite(content.extents).all() or frame.width<=0 or frame.height<=0:
        return False
    pad=renderer.points_to_pixels(5)
    p=figure.subplotpars
    left=p.left+max(0,frame.x0+pad-content.x0)/frame.width
    right=p.right-max(0,content.x1+pad-frame.x1)/frame.width
    bottom=p.bottom+max(0,frame.y0+pad-content.y0)/frame.height
    top=p.top-max(0,content.y1+pad-frame.y1)/frame.height
    if right-left<.15 or top-bottom<.15: return False
    if max(abs(left-p.left),abs(right-p.right),abs(bottom-p.bottom),abs(top-p.top))<.002:
        return False
    figure.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
    return True


class DecorationDrag:
    """Optional left-button editing mode. Only decorations are draggable."""
    def __init__(self,toolbar):
        self.toolbar=toolbar; self.window=toolbar.workbench; self.canvas=toolbar.canvas
        self.active=None
        for event,handler in [('button_press_event',self.press),('motion_notify_event',self.move),
                              ('button_release_event',self.release)]:
            self.canvas.mpl_connect(event,handler)

    def boxes(self,renderer):
        if not self.window.figure.axes: return {}
        ax=self.window.figure.axes[0]
        items={'title':[ax.title],'xlabel':[ax.xaxis.label],'ylabel':[ax.yaxis.label],
               'scale':getattr(ax,'_map_scale_artists',[]),
               'north':getattr(ax,'_map_north_artists',[])}
        boxes={key:artist_bbox(artists,renderer) for key,artists in items.items()}
        for cax in self.window.figure.axes[1:]:
            if hasattr(cax,'_colorbar'):
                boxes['colorbar_'+cax._colorbar.orientation]=cax.get_tightbbox(renderer)
        return {key:box for key,box in boxes.items() if box is not None}

    def press(self,event):
        if not self.toolbar.edit_layout.isChecked() or self.toolbar.mode or event.button!=MouseButton.LEFT:
            return
        if event.x is None or event.y is None: return
        boxes=self.boxes(self.canvas.get_renderer())
        candidates=[(key,box) for key,box in boxes.items()
                    if box.padded(4).contains(event.x,event.y)]
        if not candidates: return
        key,box=min(candidates,key=lambda item:item[1].width*item[1].height)
        initial=np.asarray(self.window.map_offsets.get(key,(0.,0.)),dtype=float)
        self.active=(key,np.array([event.x,event.y]),initial,box.frozen())
        self.window.statusBar().showMessage('Перемещение элемента оформления')

    def move(self,event):
        if self.active is None or event.x is None or event.y is None: return
        if not self.toolbar.edit_layout.isChecked() or self.toolbar.mode:
            self.active=None; return
        key,start,initial,box=self.active
        delta=np.array([event.x,event.y])-start
        frame=self.window.figure.bbox
        # Keep the selected object's visible bounding box inside the sheet,
        # if it fits. Oversized objects remain movable without a jump.
        for i,(lo,hi,blo,bhi) in enumerate(((frame.x0,frame.x1,box.x0,box.x1),
                                           (frame.y0,frame.y1,box.y0,box.y1))):
            if bhi-blo<=hi-lo: delta[i]=np.clip(delta[i],lo-blo,hi-bhi)
        offset=initial+delta/self.window.figure.dpi
        if np.max(np.abs(offset))<1e-9: self.window.map_offsets.pop(key,None)
        else: self.window.map_offsets[key]=offset.tolist()
        apply_text_offsets(self.window)
        self.window.map_dirty=True
        self.canvas.draw_idle()

    def release(self,event):
        if self.active is None: return
        self.move(event)
        self.active=None
        self.window.statusBar().showMessage('Положение элемента сохранено в проекте',3000)

"""Geographic graticule on projected XY axes, without resampling map data."""
import numpy as np
from pyproj import CRS,Transformer
from matplotlib.ticker import MaxNLocator


def degree_step(value,notation='decimal'):
    if notation=='decimal' or value==0: return float(value)
    whole=int(value); fraction=int(round((value-whole)*10000)); minutes,seconds=divmod(fraction,100)
    if minutes>=60 or seconds>=60: raise ValueError('В формате ГГГ.ММСС минуты и секунды должны быть меньше 60.')
    return whole+minutes/60+seconds/3600


def degree_label(value,notation):
    if notation=='decimal': return f'{value:.6f}'.rstrip('0').rstrip('.')+'°'
    sign='−' if value<0 else ''; seconds=int(round(abs(value)*3600)); degrees,seconds=divmod(seconds,3600); minutes,seconds=divmod(seconds,60)
    return f'{sign}{degrees}°{minutes:02d}′{seconds:02d}″'


def geographic_grid(crs,bounds,settings):
    xmin,xmax,ymin,ymax=bounds
    source=CRS.from_user_input(crs); geo=source.geodetic_crs
    inverse=Transformer.from_crs(source,geo,always_xy=True); forward=Transformer.from_crs(geo,source,always_xy=True)
    x=np.linspace(xmin,xmax,41); y=np.linspace(ymin,ymax,41)
    lon,lat=inverse.transform(np.r_[x,x,np.full(41,xmin),np.full(41,xmax)],np.r_[np.full(41,ymin),np.full(41,ymax),y,y])
    lon=np.array(lon); lat=np.array(lat); valid=np.isfinite(lon)&np.isfinite(lat)&(np.abs(lat)<=90)
    if not valid.all() or np.ptp(lon)>180: raise ValueError('Географическая сетка: проверьте СК и координаты карты; область пересекает разрыв проекции.')
    west,east=lon.min(),lon.max(); south,north=lat.min(),lat.max()
    def levels(lo,hi,value):
        step=degree_step(value,settings['notation'])
        if step==0: return MaxNLocator(nbins=5).tick_values(lo,hi)
        first=np.ceil(lo/step); last=np.floor(hi/step)
        if last-first>200: raise ValueError('Слишком густая географическая сетка: увеличьте шаг (не более 200 линий по оси).')
        return np.arange(first,last+1)*step
    meridians=levels(west,east,settings['longitude_step']); parallels=levels(south,north,settings['latitude_step'])
    curves=[]; xticks=[]; yticks=[]; edges=[]
    def intersections(a,b,bound):
        good=np.isfinite(a[:-1])&np.isfinite(a[1:])&np.isfinite(b[:-1])&np.isfinite(b[1:])
        good&=(b[:-1]-bound)*(b[1:]-bound)<=0
        indices=np.flatnonzero(good&(np.diff(b)!=0))
        return [a[i]+(a[i+1]-a[i])*(bound-b[i])/(b[i+1]-b[i]) for i in indices]
    for kind,values in [('longitude',meridians),('latitude',parallels)]:
        for value in values:
            a,b=(np.full(257,value),np.linspace(south,north,257)) if kind=='longitude' else (np.linspace(west,east,257),np.full(257,value))
            xx,yy=forward.transform(a,b); xx=np.asarray(xx); yy=np.asarray(yy)
            curves.append((xx,yy))
            bottom=[p for p in intersections(xx,yy,ymin) if xmin<=p<=xmax]
            left=[p for p in intersections(yy,xx,xmin) if ymin<=p<=ymax]
            if kind=='longitude': xticks.extend((p,value) for p in bottom)
            else: yticks.extend((p,value) for p in left)
            for bound in (ymin,ymax): edges.extend((p,bound) for p in intersections(xx,yy,bound) if xmin<=p<=xmax)
            for bound in (xmin,xmax): edges.extend((bound,p) for p in intersections(yy,xx,bound) if ymin<=p<=ymax)
    if settings['mode']=='crosses' and len(meridians)*len(parallels)>10000: raise ValueError('Слишком много крестов географической сетки: увеличьте шаг.')
    ll,pp=np.meshgrid(meridians if settings['mode']=='crosses' else [],parallels if settings['mode']=='crosses' else []); xx,yy=forward.transform(ll,pp)
    inside=np.isfinite(xx)&np.isfinite(yy)&(xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)
    return dict(curves=curves,xticks=sorted(set(xticks)),yticks=sorted(set(yticks)),edges=edges,crosses=(xx[inside],yy[inside]))


def draw_geography(ax,crs,settings):
    for artist in getattr(ax,'_geo_artists',[]): artist.remove()
    ax._geo_artists=[]
    geographic=settings['axes']=='geographic'; mode=settings['mode']
    if not geographic:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        from matplotlib.ticker import ScalarFormatter
        for axis in (ax.xaxis,ax.yaxis):
            formatter=ScalarFormatter(useOffset=False); formatter.set_scientific(False); axis.set_major_formatter(formatter)
        ax.set_xlabel('X, м'); ax.set_ylabel('Y, м')
    if not geographic and mode=='none': return None
    bounds=(*sorted(ax.get_xlim()),*sorted(ax.get_ylim())); data=geographic_grid(crs,bounds,settings)
    color=settings['color']; width=settings['width_microns']*72/25400
    if mode in ('solid','dotted'):
        for x,y in data['curves']: ax._geo_artists.extend(ax.plot(x,y,color=color,lw=width,ls=':' if mode=='dotted' else '-',zorder=500))
    elif mode=='crosses':
        x,y=data['crosses']; ax._geo_artists.append(ax.scatter(x,y,marker='+',s=30,color=color,linewidths=width,zorder=500))
    elif mode=='edges' and data['edges']:
        for x,y in data['edges']:
            marker='|' if y in bounds[2:] else '_'
            ax._geo_artists.extend(ax.plot([x],[y],marker=marker,ms=7,mew=width,color=color,clip_on=False,zorder=500))
    if geographic:
        for axis,ticks in ((ax.xaxis,data['xticks']),(ax.yaxis,data['yticks'])):
            axis.set_ticks([p for p,v in ticks]); axis.set_ticklabels([degree_label(v,settings['notation']) for p,v in ticks])
        ax.set_xlabel('Долгота'); ax.set_ylabel('Широта')
    ax.set_xlim(bounds[:2],emit=False); ax.set_ylim(bounds[2:],emit=False)
    return data

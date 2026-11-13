"""VTK scene with cell thresholding, slices, volume and isosurfaces; lazy imports."""
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QLabel, QSlider
from PyQt5.QtCore import Qt


def vtk_grid(result, component='amplitude'):
    import pyvista as pv
    mesh=result.mesh
    # VTK uses an upward vertical axis; preserve coordinates in exports.
    grid=pv.RectilinearGrid(mesh.east,mesh.north,-mesh.depth[::-1])
    m=result.model
    if m.ndim==2:
        values=np.linalg.norm(m,axis=1) if component=='amplitude' else m[:,{'East':0,'North':1,'Down':2}[component]]
        grid.cell_data['M_END']=m.reshape(mesh.shape+(3,))[:,:,::-1,:].reshape(-1,3,order='F')
    else:
        values=m
    grid.cell_data['property']=values.reshape(mesh.shape)[:,:,::-1].ravel(order='F')
    return grid


class ModelView(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QVBoxLayout(self)
        self.info=QLabel('Для интерактивного 3D нажмите «Открыть / обновить 3D». Требуется geogarmodel[volume].')
        self.info.setWordWrap(True); layout.addWidget(self.info)
        self.controls=QHBoxLayout(); layout.addLayout(self.controls)
        self.kind=QComboBox(); self.kind.addItems(['Воксели','Объём','Изоповерхности','Срезы XYZ','Интерактивное отсечение'])
        self.component=QComboBox(); self.component.addItems(['amplitude','East','North','Down'])
        self.threshold=QDoubleSpinBox(); self.threshold.setRange(0,99); self.threshold.setValue(20); self.threshold.setSuffix('% от max |m|')
        for w in (self.kind,self.component,self.threshold): self.controls.addWidget(w)
        slices=QHBoxLayout(); layout.addLayout(slices); self.slices=[]
        for name in ('East','North','Up'):
            slices.addWidget(QLabel(name))
            slider=QSlider(Qt.Horizontal); slider.setRange(0,100); slider.setValue(50)
            slider.sliderReleased.connect(self.refresh_if_ready)
            slices.addWidget(slider); self.slices.append(slider)
        self.plotter=None; self.result=None
        for w in (self.kind,self.component): w.currentTextChanged.connect(self.refresh_if_ready)
        self.threshold.valueChanged.connect(self.refresh_if_ready)

    def refresh_if_ready(self):
        if self.result is not None and self.plotter is not None:
            self.render(self.result)

    def render(self,result):
        try:
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise ValueError('Установите пакет визуализации: pip install -e ".[volume]"') from exc
        if self.plotter is None:
            self.plotter=QtInteractor(self)
            self.layout().addWidget(self.plotter.interactor,1)
        self.result=result
        p=self.plotter
        camera=p.camera_position if p.renderer.actors else None
        p.clear(); p.clear_plane_widgets()
        p.set_background('#101c2f')
        grid=vtk_grid(result,self.component.currentText())
        # Use a local origin only in the display; exported VTK retains full CRS coordinates.
        origin=np.array([result.mesh.east[0],result.mesh.north[0],0.])
        grid.translate(-origin,inplace=True)
        mode=result.settings['mode']
        unit='g/cm³' if mode=='gravity' else ('SI' if mode=='induced' else 'A/m')
        self.info.setText(f'{result.mesh.size:,} ячеек · {unit} · E₀={origin[0]:.1f}, N₀={origin[1]:.1f} м; Z вверх. Вращение: ЛКМ, масштаб: колесо.')
        self.component.setEnabled(result.model.ndim==2)
        values=grid['property']; limit=max(float(np.max(np.abs(values))),1e-15)
        cmap='viridis' if np.min(values)>=0 else 'coolwarm'
        clim=(0,limit) if np.min(values)>=0 else (-limit,limit)
        options=dict(scalars='property',cmap=cmap,clim=clim,scalar_bar_args={
            'title':unit,'color':'#dbe7f6','title_font_size':14,'label_font_size':11,
            'n_labels':4,'fmt':'%.3g','position_x':.15,'position_y':.03,'width':.65,'height':.07})
        kind=self.kind.currentIndex()
        if kind==0:
            chosen=grid.extract_cells(np.flatnonzero(np.abs(values)>=limit*self.threshold.value()/100))
            if chosen.n_cells: p.add_mesh(chosen,show_edges=True,edge_color='#33465a',**options)
        elif kind==1:
            p.add_volume(grid.cell_data_to_point_data(),opacity='sigmoid',**options)
        elif kind==2:
            points=grid.cell_data_to_point_data()
            lo,hi=float(values.min()),float(values.max())
            if hi>lo:
                surface=points.contour(np.linspace(lo,hi,7)[1:-1],scalars='property')
                if surface.n_points: p.add_mesh(surface,opacity=.65,**options)
        elif kind==3:
            centers=[grid.bounds[2*i]+(grid.bounds[2*i+1]-grid.bounds[2*i])*slider.value()/100 for i,slider in enumerate(self.slices)]
            p.add_mesh(grid.slice_orthogonal(x=centers[0],y=centers[1],z=centers[2]),**options)
            p.add_mesh(grid.outline(),color='#93a9c4')
        else:
            p.add_mesh_clip_plane(grid,normal='x',**options)
        p.add_axes(xlabel='East',ylabel='North',zlabel='Up')
        p.show_grid(color='#93a9c4',font_size=10,bold=False,fmt='%.0f',use_2d=True,
                    xtitle='East (m)',ytitle='North (m)',ztitle='Up (m)',n_xlabels=3,n_ylabels=3,n_zlabels=3)
        if camera is None: p.view_isometric(); p.reset_camera()
        else: p.camera_position=camera
        p.render()

    def shutdown(self):
        if self.plotter is not None:
            self.plotter.close()

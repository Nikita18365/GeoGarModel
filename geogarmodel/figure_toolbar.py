"""Export a detached Matplotlib figure without rendering a Qt canvas in a worker."""
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from .activity_ui import find_activity


class FigureToolbar(NavigationToolbar2QT):
    def save_figure(self,*args):
        activity=find_activity(self)
        if activity is None:
            return super().save_figure(*args)
        types=self.canvas.get_supported_filetypes_grouped()
        filters={f"{name} ({' '.join('*.'+ext for ext in exts)})":exts for name,exts in sorted(types.items())}
        default=next(key for key,exts in filters.items() if self.canvas.get_default_filetype() in exts)
        directory=mpl.rcParams['savefig.directory']
        start=Path(directory).expanduser()/self.canvas.get_default_filename()
        path,selected=QFileDialog.getSaveFileName(self,'Сохранить изображение',str(start),';;'.join(filters),default)
        if not path:
            return
        if not Path(path).suffix:
            path+='.'+filters[selected or default][0]
        try:
            snapshot=deepcopy(self.canvas.figure)
            FigureCanvasAgg(snapshot)
            activity.run('Экспорт изображения',snapshot.savefig,path)
            if directory:
                mpl.rcParams['savefig.directory']=str(Path(path).parent)
        except Exception as error:
            QMessageBox.warning(self,'Экспорт изображения',str(error))
            return
        return path

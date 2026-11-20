"""Shared status-bar progress and a GUI-responsive bridge for synchronous file actions."""
from contextlib import contextmanager

from PyQt5.QtCore import QObject, QThread, QEventLoop, QEvent, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QProgressBar
from .work_progress import reporting


def find_activity(widget):
    while widget is not None:
        if hasattr(widget,'activity'):
            return widget.activity
        widget=widget.parentWidget()
    return None


class FileWorker(QThread):
    update = pyqtSignal(str, int)

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.value = self.error = None

    def run(self):
        try:
            with reporting(self.update.emit):
                self.value = self.function()
        except Exception as error:
            self.error = error
        finally:
            self.function = None


class Activities(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.panel = QWidget()
        row = QHBoxLayout(self.panel)
        row.setContentsMargins(4, 0, 4, 0)
        self.bar = QProgressBar()
        self.bar.setFixedSize(150, 19)
        self.bar.setAlignment(Qt.AlignCenter)
        self.bar.setAccessibleName('Ход операции')
        self.bar.setStyleSheet('QProgressBar { border: 1px solid #526579; border-radius: 4px; '
                               'background: #182735; color: white; text-align: center; } '
                               'QProgressBar::chunk { background: #23854e; }')
        self.label = QLabel()
        row.addWidget(self.bar)
        row.addWidget(self.label)
        row.addStretch()
        window.statusBar().insertPermanentWidget(0, self.panel, 1)
        self.panel.hide()
        self.jobs = {}
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.panel.hide)
        self.waiting = 0
        self.cancel_dialogs = []
        self.modal_at_start = None

    def start(self, title):
        key = object()
        self.hide_timer.stop()
        self.jobs[key] = (title, '', -1)
        self.display()
        return key

    def update(self, key, stage, percent):
        if key in self.jobs:
            self.jobs[key] = (self.jobs[key][0], stage, percent)
            if key is next(reversed(self.jobs)):
                self.display()

    def display(self):
        title, stage, percent = next(reversed(self.jobs.values()))
        self.label.setText(title + (' · ' + stage if stage else ''))
        self.bar.setRange(0, 0 if percent < 0 else 100)
        if percent >= 0:
            self.bar.setValue(percent)
        self.panel.show()

    def finish(self, key, outcome='Готово'):
        if key not in self.jobs:
            return
        title, _, percent = self.jobs.pop(key)
        if self.jobs:
            self.display()
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(100 if outcome=='Готово' else max(0,percent))
            self.label.setText(title + ' · ' + outcome)
            self.hide_timer.start(2000)

    @contextmanager
    def immediate(self, title):
        """Mark a small atomic GUI action without dispatching queued trial events."""
        key = self.start(title)
        try:
            yield
        except Exception:
            self.finish(key, 'Ошибка')
            raise
        else:
            self.finish(key)

    def track(self, worker, title):
        key = self.start(title)
        worker.activity_key = key
        worker.activity_outcome = 'Готово'
        worker.phase.connect(lambda stage, percent: self.update(key, stage, percent))
        worker.failed.connect(lambda _: setattr(worker, 'activity_outcome', 'Ошибка'))
        if hasattr(worker, 'cancelled'):
            worker.cancelled.connect(lambda: setattr(worker, 'activity_outcome', 'Остановлено'))
        worker.delivering = False
        worker.completed = False
        def finished():
            worker.completed = True
            if not worker.delivering:
                self.finish(key, worker.activity_outcome)
        worker.finished.connect(finished)

    def deliver(self, worker, callback, result):
        worker.delivering = True
        try:
            callback(result)
        except Exception:
            worker.activity_outcome = 'Ошибка'
            raise
        finally:
            worker.delivering = False
            if worker.completed:
                self.finish(worker.activity_key, worker.activity_outcome)

    def eventFilter(self, watched, event):
        # A synchronous caller still owns its inputs and its destination window.
        # Keep paint, timers, resizing and worker signals flowing during its I/O.
        if self.waiting and event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                QEvent.MouseButtonDblClick, QEvent.KeyPress, QEvent.KeyRelease, QEvent.Shortcut,
                QEvent.Wheel, QEvent.Drop, QEvent.TouchBegin, QEvent.Close):
            if isinstance(watched,QWidget) and any(watched is d or d.isAncestorOf(watched) for d in self.cancel_dialogs):
                return False
            modal = QApplication.activeModalWidget()
            if modal is not None and modal is not self.modal_at_start and isinstance(watched,QWidget):
                if watched is modal or modal.isAncestorOf(watched):
                    return False
            if event.type()==QEvent.Close:
                event.ignore()
            return True
        return False

    def run(self, title, function, *args, **kwargs):
        return self.file_job(title,lambda: function(*args, **kwargs))

    def file_job(self,title,function,dialog=None):
        worker = FileWorker(function)
        key = self.start(title)
        loop = QEventLoop()
        worker.update.connect(lambda stage, percent: self.update(key, stage, percent), Qt.QueuedConnection)
        worker.finished.connect(loop.quit, Qt.QueuedConnection)
        app = QApplication.instance()
        if not self.waiting:
            self.modal_at_start = app.activeModalWidget()
            app.installEventFilter(self)
        if dialog is not None:
            self.cancel_dialogs.append(dialog)
            worker.update.connect(lambda stage,pct: dialog.setValue(max(0,pct)),Qt.QueuedConnection)
        self.waiting += 1
        try:
            worker.start()
            loop.exec_()
            worker.wait()
            if worker.error is not None:
                raise worker.error
            return worker.value
        finally:
            if dialog is not None:
                self.cancel_dialogs.remove(dialog)
            self.waiting -= 1
            if not self.waiting:
                app.removeEventFilter(self)
                self.modal_at_start = None
            self.finish(key, 'Ошибка' if worker.error is not None else 'Остановлено' if dialog is not None and dialog.wasCanceled() else 'Готово')

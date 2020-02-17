from IPython.display import display
import ipywidgets as widgets
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper
import threading
import abc

class Solver(object):
    def calculate(self):
        pass
    def reset(self):
        pass
        
class ControlPanel(object):
    def __init__(self, solver, plotter):
        self._solver = solver
        self._plotter = plotter
        #
        self.counter_int = widgets.IntText(value=0, description='Current Count:', disabled=True)
        self.counter_limit_int = widgets.IntText(value=1000, description='Count Limit:', disabled=False)
        display(widgets.HBox([self.counter_int, self.counter_limit_int]))
        #
        self.current_time = widgets.FloatText(value=0, description='Current Time:', disabled=True)
        self.plot_every_int = widgets.IntText(value=10, description='Plot Every:', disabled=False)
        display(widgets.HBox([self.current_time, self.plot_every_int]))
        #
        self.calc_button = widgets.Button(description='Calculate')
        self.pause_button = widgets.Button(description='Pause')
        self.reset_button = widgets.Button(description='Reset')
        display(widgets.HBox([self.calc_button, self.pause_button, self.reset_button]))
        #
        self.calc_button.on_click(self.calculate)
        self.pause_button.on_click(self.pause)
        self.reset_button.on_click(self.reset)
        #
        self._evt_pause = threading.Event()
        self._evt_pause.clear()
        self._thread = threading.Thread(target=self.calc_thread)
        self._thread.start()
    def calc_thread(self):
        while self.counter_int.value < self.counter_limit_int.value:
            self._evt_pause.wait()
            self.counter_int.value += 1
            self.current_time.value = self._solver.timeSeries.current_time
            self._solver.calculate()
            if not (self.counter_int.value % self.plot_every_int.value):
                self._plotter.push()
    def calculate(self, e):
        self._evt_pause.set()
    def pause(self, e):
        self._evt_pause.clear()
    def reset(self, e):
        self.pause(e)
        self.counter_int.value = 0
        self._solver.reset()
        self.current_time.value = self._solver.timeSeries.current_time
        self._plotter.yee = self._solver.yee
        self._plotter.push()
        self.thread = threading.Thread(target=self.calc_thread)
        self.thread.start()
        
class Plotter(object):
    def __init__(self, yee, slices, mapper):
        self.yee = y = yee
        self._slices = s = slices
        ex, ey, ez = y.e
        im = ez.__getitem__(s)
        output_notebook()
        f = figure()
        self._render = f.image(image=[im], x=0, y=0, dw=10, dh=10, color_mapper=mapper)
        self._notebook_handle = show(f, notebook_handle=True)
    def push(self):
        ex, ey, ez = self.yee.e
        self._render.data_source.data['image'] = [ez.__getitem__(self._slices)]
        push_notebook(handle=self._notebook_handle)

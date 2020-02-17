from IPython.display import display
import ipywidgets as widgets
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper
import threading

class UI(object):
    def __init__(self, solver=None):
        if solver:
            self._set_solver(solver)
            
        self._evt_pause = threading.Event()
        self._evt_pause.clear()
        
        self.counter_int = widgets.IntText(value=0, description='Current Count:', disabled=True)
        self.counter_limit_int = widgets.IntText(value=1000, description='Count Limit:', disabled=False)
        display(widgets.HBox([self.counter_int, self.counter_limit_int]))
        
        self.plot_every_int = widgets.IntText(value=10, description='Plot Every:', disabled=False)
        display(self.plot_every_int)

        self.calc_button = widgets.Button(description='Calculate')
        self.calc_button.on_click(self.calculate)
        self.pause_button = widgets.Button(description='Pause')
        self.pause_button.on_click(self.pause)
        self.reset_button = widgets.Button(description='Reset')
        self.reset_button.on_click(self.reset)
        display(widgets.HBox([self.calc_button, self.pause_button, self.reset_button]))
        
        output_notebook()
        self.figure = figure()
        self.mapper = None
        self.slices = None
        self._render = None
        self._notebook_handle = None
        
        self.thread = threading.Thread(target=self.calc_thread)
        self.thread.start()
    def _set_solver(self, solver):
        self._solver = s = solver
    solver = property(lambda self: self._solver, _set_solver)
    def calc_thread(self):
        while self.counter_int.value < self.counter_limit_int.value:
            self._evt_pause.wait()
            self.counter_int.value += 1
            self.solver.calculate()
            if not (self.counter_int.value % self.plot_every_int.value):
                self.push_notebook()
    def calculate(self, e):
        self._evt_pause.set()
    def pause(self, e):
        self._evt_pause.clear()
    def reset(self, e):
        self.pause(e)
        self.counter_int.value = 0
        self.solver.reset()
        self.thread = threading.Thread(target=self.calc_thread)
        self.thread.start()
    def render(self):
        ex, ey, ez = self._solver.yee.e
        self._render = self.figure.image(image=[ez.__getitem__(self.slices)], x=0, y=0, dw=10, dh=10, color_mapper=self.mapper)
        self._notebook_handle = show(self.figure, notebook_handle=True)
    def push_notebook(self):
        ex, ey, ez = self._solver.yee.e
        self._render.data_source.data['image'] = [ez.__getitem__(self.slices)]
        push_notebook(handle=self._notebook_handle)

import pyfdtd3d.structuredpoints_cupy as fdtd
import pyfdtd3d.structuredpoints_ui as ui
from pyfdtd3d.base import sin3, vsrc
import time

class MySolver(ui.Solver):
    def __init__(self, **kw):
        ui.Solver.__init__(self, **kw)
    def prepare(self):
        p = self.parameters
        self.time = t = fdtd.TimeSeries(p['dt']) # [second]
        self.grid = g = fdtd.Grid(
            dimensions=p['dimensions'], # [cells]
            spacing=p['spacing'], # [mm]
        )
        self.yee = y = fdtd.Yee(grid=g)
        self.material = m = fdtd.Material(*g.dimensions)
        self.hcalc = fdtd.HCalc(yee=y, time=t)
        self.ecalc = fdtd.ECalc(yee=y, time=t, material=m)
        self.currentz = fdtd.CurrentZ(yee=y, pos=p['pos']) # <-- fdtd.CurrentZ(yee=y)
    def calculate(self):
        y, t, hc, ec, cz = self.yee, self.time, self.hcalc, self.ecalc, self.currentz
        dx, dy, dz = y.grid.spacing
        freq = self.parameters['freq']
        t.progress()
        hc.calculate()
        t.progress()
        ec.calculate()
        y.ez[50, 50, 26:37] = 0.
        y.ez[50, 50, 14:25] = 0.
        y.ez[50, 50, 25] = vsrc(voltage=sin3(t.current_time, freq), current=cz.get(), delta=dz) # <-- cz[50, 50, 25]

s = MySolver(
    dt=0.01e-9, # [second]
    dimensions=(100, 100, 51), # [cells]
    spacing=(6e-3, 6e-3, 6e-3), # [mm]
    pos=(50, 50, 25),
    freq=1e+9, # [Hz]
)
s.prepare()

p = ui.Plotter(
    yee=s.yee,
    slices=(50, slice(None), slice(None)),
    mapper=ui.LinearColorMapper(palette='Viridis256', low=-1, high=1),
)

u = ui.ControlPanel(p, s)

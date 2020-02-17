import numpy as np

class EventHandler(object):
    def __init__(self):
        self._hook = []
    def hook(self, func):
        self._hook.append(func)
    def dispatch(self):
        for func in self._hook:
            func()

class Grid(object):
    def __init__(self):
        self.events = {}

class TimeSeries(object):
    def __init__(self):
        self.current_time = 0.
        self.dt = 1 # [ns]
    def progress_half_step(self):
        self.current_time += .5 * self.dt

def sin3(time,freq): # time: TimeSeries, freq: float
    omg0 = 2 * np.pi * freq # 正弦波源の各周波数
    tau0 = .5 / freq        # パルス持続半周期
    if time.current_time > 2. * tau0:
        v = 0.
    else:
        v = np.sin(omg0 * time.current_time) ** 3
    return v

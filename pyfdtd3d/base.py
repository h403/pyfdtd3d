import numpy as np

class Grid(object):
    pass

class Calculator(object):
    def calculate(self):
        pass

class Probe(object):
    def get(self):
        pass
    
class TimeSeries(object):
    dt = property(lambda self: self._dt)
    ratio = property(lambda self: self._ratio)
    current_time = property(lambda self: self._current_time)
    def __init__(self, dt, ratio=.5):
        self._current_time = 0.
        self._dt = dt # [ns]
        self._ratio = ratio
    def progress(self):
        self._current_time += self.ratio * self.dt

def sin3(current_time,freq): # time: float, freq: float
    omg0 = 2 * np.pi * freq # 正弦波源の各周波数
    tau0 = .5 / freq        # パルス持続半周期
    if current_time > 2. * tau0:
        v = 0.
    else:
        v = np.sin(omg0 * current_time) ** 3
    return v

def sin(current_time,freq): # time: float, freq: float
    omg0 = 2 * np.pi * freq # 正弦波源の各周波数
    return np.sin(omg0 * current_time) ** 3

def vsrc(voltage, current, delta, resister=50):
    return - (voltage - resister * current) / delta

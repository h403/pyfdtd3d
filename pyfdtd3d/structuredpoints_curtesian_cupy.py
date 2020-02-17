from pyfdtd3d.structuredpoints_curtesian import *
import cupy as cp

class FieldCP(Field):
    def alloc(self, nx, ny, nz):
        self.value = cp.zeros((nx, ny, nz))

class FieldsCP(Fields):
    def __init__(self):
        Fields.__init__(self)
        self.values = dict([(k, FieldCP()) for k in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']])

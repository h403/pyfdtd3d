from pyfdtd3d.constants import EP0, MU0
from pyfdtd3d.base import TimeSeries
from pyfdtd3d.structuredpoints import Grid
import numpy as np
import gc

class Field(object):
    def __init__(self):
        self.value = None
    def alloc(self, nx, ny, nz):
        self.value = np.zeros((nx, ny, nz))
    def free(self):
        self.value = None
        gc.collect()
        
class Fields(object):
    ex = property(lambda self: self.values['ex'].value)
    ey = property(lambda self: self.values['ey'].value)
    ez = property(lambda self: self.values['ez'].value)
    hx = property(lambda self: self.values['hx'].value)
    hy = property(lambda self: self.values['hy'].value)
    hz = property(lambda self: self.values['hz'].value)
    def __init__(self):
        self.dimensions = None, None, None
        self.values = dict([(k, Field()) for k in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']])
    def alloc(self, nx, ny, nz):
        self.dimensions = nx, ny, nz
        self.values['ex'].alloc(nx, ny+1, nz+1)
        self.values['ey'].alloc(nx+1, ny, nz+1)
        self.values['ez'].alloc(nx+1, ny+1, nz)
        self.values['hx'].alloc(nx+1, ny, nz)
        self.values['hy'].alloc(nx, ny+1, nz)
        self.values['hz'].alloc(nx, ny, nz+1)
    def free(self):
        for v in self.values.values():
            v.free()

class SetDimensionsListner(object):
    def __init__(self, grid=None):
        if grid:
            self._set_grid(grid)
    def _set_grid(self, grid):
        self._grid = g = grid
        g.events['set_dimensions'].hook(self._set_dimensions)
    def _set_dimensions(self):
        self.alloc(*self._grid.dimensions)
    grid = property(lambda self: self._grid, _set_grid)
    
class Yee(SetDimensionsListner):
    def __init__(self, grid=None):
        SetDimensionsListner.__init__(self, grid)
        self._fields = Fields()
    def alloc(self, nx, ny, nz):
        self._fields.alloc(nx, ny, nz)
    def free(self):
        self._fields.free()
    e = property(lambda self: (self._fields.ex, self._fields.ey, self._fields.ez))
    h = property(lambda self: (self._fields.hx, self._fields.hy, self._fields.hz))
    
class Material(SetDimensionsListner):
    def __init__(self, grid=None):
        SetDimensionsListner.__init__(self, grid)
        self._field = Field()
    def alloc(self, nx, ny, nz):
        self._field.alloc(nx, ny, nz)
        self._field.value[:, :, :] = 1.
    def free(self):
        self._field.free()
    er = property(lambda self: self._field.value)

class Solver(object):
    def __init__(self):
        self.grid = g = Grid()
        self.material = Material(g)
        self.yee = Yee(g)
        self.time = TimeSeries()
    def prepare(self):
        self.calc_er()
    def reset(self):
        pass
    def calculate(self):
        self.sequence()
    def sequence(self):
        pass
    def calc_h(self):
        ex, ey, ez = self.yee.e
        hx, hy, hz = self.yee.h
        dx, dy, dz = self.grid.spacing
        dt = self.time.dt
        hx[ :  , :  , :  ] -= dt / MU0 / dy * (ez[ :  ,1:  , :  ] - ez[ :  , :-1, :  ])\
                            - dt / MU0 / dz * (ey[ :  , :  ,1:  ] - ey[ :  , :  , :-1])
        hy[ :  , :  , :  ] -= dt / MU0 / dz * (ex[ :  , :  ,1:  ] - ex[ :  , :  , :-1])\
                            - dt / MU0 / dx * (ez[1:  , :  , :  ] - ez[ :-1, :  , :  ])
        hz[ :  , :  , :  ] -= dt / MU0 / dx * (ey[1:  , :  , :  ] - ey[ :-1, :  , :  ])\
                            - dt / MU0 / dy * (ex[ :  ,1:  , :  ] - ex[ :  , :-1, :  ])       
    def calc_e(self):
        ex, ey, ez = self.yee.e
        hx, hy, hz = self.yee.h
        dx, dy, dz = self.grid.spacing
        dt = self.time.dt
        erx, ery, erz = self._er
        ex[ :  ,1:-1,1:-1] += dt / EP0 / erx / dy * (hz[ :  ,1:  ,1:-1] - hz[ :  , :-1,1:-1])\
                            - dt / EP0 / erx / dz * (hy[ :  ,1:-1,1:  ] - hy[ :  ,1:-1, :-1])
        ey[1:-1, :  ,1:-1] += dt / EP0 / ery / dz * (hx[1:-1, :  ,1:  ] - hx[1:-1, :  , :-1])\
                            - dt / EP0 / ery / dx * (hz[1:  , :  ,1:-1] - hz[ :-1, :  ,1:-1])
        ez[1:-1,1:-1, :  ] += dt / EP0 / erz / dx * (hy[1:  ,1:-1, :  ] - hy[ :-1,1:-1, :  ])\
                            - dt / EP0 / erz / dy * (hx[1:-1,1:  , :  ] - hx[1:-1, :-1, :  ])
    def calc_er(self):
        er = self.material.er
        self._erx = (er[ :  , :-1, :-1] +
                     er[ :  , :-1,1:  ] +
                     er[ :  ,1:  , :-1] +
                     er[ :  ,1:  ,1:  ]) / 4.
        self._ery = (er[ :-1, :  , :-1] +
                     er[1:  , :  , :-1] +
                     er[ :-1, :  ,1:  ] +
                     er[1:  , :  ,1:  ]) / 4.
        self._erz = (er[ :-1, :-1, :  ] +
                     er[ :-1,1:  , :  ] +
                     er[1:  , :-1, :  ] +
                     er[1:  ,1:  , :  ]) / 4.
    _er = property(lambda self: (self._erx, self._ery, self._erz))
    def current_x(self,i,j,k):
        hx, hy, hz = self.yee.h
        dx, dy, dz = self.grid.spacing
        return (hz[i,j,k] - hz[i  ,j-1,k  ]) * dz + (hy[i  ,j  ,k-1] - hy[i,j,k]) * dy
    def current_y(self,i,j,k):
        hx, hy, hz = self.yee.h
        dx, dy, dz = self.grid.spacing
        return (hx[i,j,k] - hx[i  ,j  ,k-1]) * dx + (hz[i-1,j  ,k  ] - hz[i,j,k]) * dz
    def current_z(self,i,j,k):
        hx, hy, hz = self.yee.h
        dx, dy, dz = self.grid.spacing
        return (hy[i,j,k] - hy[i-1,j  ,k  ]) * dy + (hx[i  ,j-1,k  ] - hx[i,j,k]) * dx
    def source_z(self,i,j,k,v,r=50.):
        hx, hy, hz = self.yee.h
        ex, ey, ez = self.yee.e
        dx, dy, dz = self.grid.spacing
        c = self.current_z(i,j,k)
        ez[i,j,k] = - (v - r * c) / dz


from pyfdtd3d.constants import EP0, MU0
from pyfdtd3d.base import TimeSeries
import pyfdtd3d.base
import numpy as np
import gc

class Grid(pyfdtd3d.base.Grid):
    dimensions = property(lambda self: self._dimensions)
    spacing = property(lambda self: self._spacing)
    origin = property(lambda self: self._origin)
    def __init__(self, dimensions, spacing, origin=(0, 0, 0)):
        pyfdtd3d.base.Grid.__init__(self)
        self._dimensions = dimensions # (nx, ny, nz)
        self._spacing = spacing # (dx, dy, dz)
        self._origin = origin

class Yee(object):
    ex = property(lambda self: self._ex)
    ey = property(lambda self: self._ey)
    ez = property(lambda self: self._ez)
    hx = property(lambda self: self._hx)
    hy = property(lambda self: self._hy)
    hz = property(lambda self: self._hz)
    e = property(lambda self: (self.ex, self.ey, self.ez))
    h = property(lambda self: (self.hx, self.hy, self.hz))
    grid = property(lambda self: self._grid)
    def __init__(self, grid):
        self._grid = g = grid
        nx, ny, nz = g.dimensions
        self._ex = np.zeros((nx, ny+1, nz+1))
        self._ey = np.zeros((nx+1, ny, nz+1))
        self._ez = np.zeros((nx+1, ny+1, nz))
        self._hx = np.zeros((nx+1, ny, nz))
        self._hy = np.zeros((nx, ny+1, nz))
        self._hz = np.zeros((nx, ny, nz+1))

class Material(object):
    er = property(lambda self: self._er)
    def __init__(self, nx, ny, nz):
        self._er = np.ones((nx, ny, nz))

class HCalc(pyfdtd3d.base.Calculator):
    def __init__(self, yee, time):
        pyfdtd3d.base.Calculator.__init__(self)
        self._yee = y = yee
        dx, dy, dz = y.grid.spacing
        self._const = time.dt / MU0 / np.array((dx, dy, dz))
        self.time = time
    def calculate_(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        cx, cy, cz = self._const
        hx[ :  , :  , :  ] -= cy * (ez[ :  ,1:  , :  ] - ez[ :  , :-1, :  ])\
                            - cz * (ey[ :  , :  ,1:  ] - ey[ :  , :  , :-1])
        hy[ :  , :  , :  ] -= cz * (ex[ :  , :  ,1:  ] - ex[ :  , :  , :-1])\
                            - cx * (ez[1:  , :  , :  ] - ez[ :-1, :  , :  ])
        hz[ :  , :  , :  ] -= cx * (ey[1:  , :  , :  ] - ey[ :-1, :  , :  ])\
                            - cy * (ex[ :  ,1:  , :  ] - ex[ :  , :-1, :  ])       
    def calculate(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        dt = self.time.dt
        hx[ :  , :  , :  ] -= dt / MU0 / dy * (ez[ :  ,1:  , :  ] - ez[ :  , :-1, :  ])\
                            - dt / MU0 / dz * (ey[ :  , :  ,1:  ] - ey[ :  , :  , :-1])
        hy[ :  , :  , :  ] -= dt / MU0 / dz * (ex[ :  , :  ,1:  ] - ex[ :  , :  , :-1])\
                            - dt / MU0 / dx * (ez[1:  , :  , :  ] - ez[ :-1, :  , :  ])
        hz[ :  , :  , :  ] -= dt / MU0 / dx * (ey[1:  , :  , :  ] - ey[ :-1, :  , :  ])\
                            - dt / MU0 / dy * (ex[ :  ,1:  , :  ] - ex[ :  , :-1, :  ])       

class ECalc(pyfdtd3d.base.Calculator):
    def __init__(self, yee, time, material):
        pyfdtd3d.base.Calculator.__init__(self)
        self._yee = y = yee
        dx, dy, dz = y.grid.spacing
        er = material.er
        erx = (er[ :  , :-1, :-1] +
               er[ :  , :-1,1:  ] +
               er[ :  ,1:  , :-1] +
               er[ :  ,1:  ,1:  ]) / 4.
        ery = (er[ :-1, :  , :-1] +
               er[1:  , :  , :-1] +
               er[ :-1, :  ,1:  ] +
               er[1:  , :  ,1:  ]) / 4.
        erz = (er[ :-1, :-1, :  ] +
               er[ :-1,1:  , :  ] +
               er[1:  , :-1, :  ] +
               er[1:  ,1:  , :  ]) / 4.
        e = np.array((erx, erx, ery, ery, erz, erz))
        d = np.array((dy, dz, dz, dx, dx, dy))
        self._const = time.dt / EP0 / e / d # cxy, cxz, cyz, cyx, czx, czy
        self.time = time
        self._er = erx, ery, erz
    def calculate_(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        cxy, cxz, cyz, cyx, czx, czy = self._const
        ex[ :  ,1:-1,1:-1] += cxy * (hz[ :  ,1:  ,1:-1] - hz[ :  , :-1,1:-1])\
                            - cxz * (hy[ :  ,1:-1,1:  ] - hy[ :  ,1:-1, :-1])
        ey[1:-1, :  ,1:-1] += cyz * (hx[1:-1, :  ,1:  ] - hx[1:-1, :  , :-1])\
                            - cyx * (hz[1:  , :  ,1:-1] - hz[ :-1, :  ,1:-1])
        ez[1:-1,1:-1, :  ] += czx * (hy[1:  ,1:-1, :  ] - hy[ :-1,1:-1, :  ])\
                            - czy * (hx[1:-1,1:  , :  ] - hx[1:-1, :-1, :  ])
    def calculate(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        dt = self.time.dt
        erx, ery, erz = self._er
        ex[ :  ,1:-1,1:-1] += dt / EP0 / erx / dy * (hz[ :  ,1:  ,1:-1] - hz[ :  , :-1,1:-1])\
                            - dt / EP0 / erx / dz * (hy[ :  ,1:-1,1:  ] - hy[ :  ,1:-1, :-1])
        ey[1:-1, :  ,1:-1] += dt / EP0 / ery / dz * (hx[1:-1, :  ,1:  ] - hx[1:-1, :  , :-1])\
                            - dt / EP0 / ery / dx * (hz[1:  , :  ,1:-1] - hz[ :-1, :  ,1:-1])
        ez[1:-1,1:-1, :  ] += dt / EP0 / erz / dx * (hy[1:  ,1:-1, :  ] - hy[ :-1,1:-1, :  ])\
                            - dt / EP0 / erz / dy * (hx[1:-1,1:  , :  ] - hx[1:-1, :-1, :  ])


class CurrentX(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        pyfdtd3d.base.Probe.__init__(self)
        self._yee = yee
    def get(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hz[i,j,k] - hz[i  ,j-1,k  ]) * dz + (hy[i  ,j  ,k-1] - hy[i,j,k]) * dy

class CurrentY(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        pyfdtd3d.base.Probe.__init__(self)
        self._yee = yee
    def __getitem__(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hx[i,j,k] - hx[i  ,j  ,k-1]) * dx + (hz[i-1,j  ,k  ] - hz[i,j,k]) * dz

class CurrentZ(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        pyfdtd3d.base.Probe.__init__(self)
        self._yee = yee
    def __getitem__(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hy[i,j,k] - hy[i-1,j  ,k  ]) * dy + (hx[i  ,j-1,k  ] - hx[i,j,k]) * dx
        

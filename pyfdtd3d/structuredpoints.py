from pyfdtd3d.constants import EP0, MU0
from pyfdtd3d.base import TimeSeries
import pyfdtd3d.base
import numpy as np
import gc

class Grid(pyfdtd3d.base.Grid):
    dimensions = property(lambda self: self._dimensions)
    nx = property(lambda self: self._dimensions[0])
    ny = property(lambda self: self._dimensions[1])
    nz = property(lambda self: self._dimensions[2])
    spacing = property(lambda self: self._spacing)
    dx = property(lambda self: self._spacing[0])
    dy = property(lambda self: self._spacing[1])
    dz = property(lambda self: self._spacing[2])
    origin = property(lambda self: self._origin)
    def __init__(self, dimensions, spacing, origin=(0, 0, 0)):
        super().__init__()
        self._dimensions = dimensions # (nx, ny, nz)
        self._spacing = spacing # (dx, dy, dz)
        self._origin = origin

class HField(object):
    hx = property(lambda self: self._hx)
    hy = property(lambda self: self._hy)
    hz = property(lambda self: self._hz)
    def __init__(self, nx, ny, nz):
        self._hx = np.zeros((nx+1, ny, nz))
        self._hy = np.zeros((nx, ny+1, nz))
        self._hz = np.zeros((nx, ny, nz+1))
        
class EField(object):
    ex = property(lambda self: self._ex)
    ey = property(lambda self: self._ey)
    ez = property(lambda self: self._ez)
    def __init__(self, nx, ny, nz):
        self._ex = np.zeros((nx, ny+1, nz+1))
        self._ey = np.zeros((nx+1, ny, nz+1))
        self._ez = np.zeros((nx+1, ny+1, nz))
        
class Yee(object):
    ex = property(lambda self: self._efield.ex)
    ey = property(lambda self: self._efield.ey)
    ez = property(lambda self: self._efield.ez)
    hx = property(lambda self: self._hfield.hx)
    hy = property(lambda self: self._hfield.hy)
    hz = property(lambda self: self._hfield.hz)
    e = property(lambda self: (self.ex, self.ey, self.ez))
    h = property(lambda self: (self.hx, self.hy, self.hz))
    grid = property(lambda self: self._grid)
    def __init__(self, grid):
        self._grid = g = grid
        self._efield = EField(*grid.dimensions)
        self._hfield = HField(*grid.dimensions)
        
class Material(object):
    er = property(lambda self: self._er)
    def __init__(self, nx, ny, nz):
        self._er = np.ones((nx, ny, nz))

class HCalc(pyfdtd3d.base.Calculator):
    def __init__(self, yee, time):
        super().__init__()
        self._yee = y = yee
        self.time = t = time
        self._const = t.dt / MU0 / np.array(y.grid.spacing)
    def calculate(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        cx, cy, cz = self._const
        hx[ :  , :  , :  ] -= cy * (ez[ :  ,1:  , :  ] - ez[ :  , :-1, :  ])\
                            - cz * (ey[ :  , :  ,1:  ] - ey[ :  , :  , :-1])
        hy[ :  , :  , :  ] -= cz * (ex[ :  , :  ,1:  ] - ex[ :  , :  , :-1])\
                            - cx * (ez[1:  , :  , :  ] - ez[ :-1, :  , :  ])
        hz[ :  , :  , :  ] -= cx * (ey[1:  , :  , :  ] - ey[ :-1, :  , :  ])\
                            - cy * (ex[ :  ,1:  , :  ] - ex[ :  , :-1, :  ])       

class ECalc(pyfdtd3d.base.Calculator):
    def __init__(self, yee, time, material):
        super().__init__()
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
        self._er = erx, ery, erz
        e = np.array((erx, erx, ery, ery, erz, erz))
        d = np.array((dy, dz, dz, dx, dx, dy))
        self._const = time.dt / EP0 / e / d # cxy, cxz, cyz, cyx, czx, czy
    def calculate(self):
        ex, ey, ez = self._yee.e
        hx, hy, hz = self._yee.h
        cxy, cxz, cyz, cyx, czx, czy = self._const
        ex[ :  ,1:-1,1:-1] += cxy * (hz[ :  ,1:  ,1:-1] - hz[ :  , :-1,1:-1])\
                            - cxz * (hy[ :  ,1:-1,1:  ] - hy[ :  ,1:-1, :-1])
        ey[1:-1, :  ,1:-1] += cyz * (hx[1:-1, :  ,1:  ] - hx[1:-1, :  , :-1])\
                            - cyx * (hz[1:  , :  ,1:-1] - hz[ :-1, :  ,1:-1])
        ez[1:-1,1:-1, :  ] += czx * (hy[1:  ,1:-1, :  ] - hy[ :-1,1:-1, :  ])\
                            - czy * (hx[1:-1,1:  , :  ] - hx[1:-1, :-1, :  ])

class CurrentX(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        super().__init__()
        self._yee = yee
    def __getitem__(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hz[i,j,k] - hz[i  ,j-1,k  ]) * dz + (hy[i  ,j  ,k-1] - hy[i,j,k]) * dy

class CurrentY(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        super().__init__()
        self._yee = yee
    def __getitem__(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hx[i,j,k] - hx[i  ,j  ,k-1]) * dx + (hz[i-1,j  ,k  ] - hz[i,j,k]) * dz

class CurrentZ(pyfdtd3d.base.Probe):
    def __init__(self, yee):
        super().__init__()
        self._yee = yee
    def __getitem__(self, pos):
        hx, hy, hz = self._yee.h
        dx, dy, dz = self._yee.grid.spacing
        i, j, k = pos
        return (hy[i,j,k] - hy[i-1,j  ,k  ]) * dy + (hx[i  ,j-1,k  ] - hx[i,j,k]) * dx


class PML(object):
    @classmethod
    def sigma(cls, delta, thickness=16, r0=1e-6, m=2):
        smax = - ((m + 1.) * EP0 / np.sqrt(EP0 * MU0)) / (2. * thickness * delta) * np.log(r0)
        pos  = np.linspace(0, thickness * delta, thickness + 1, endpoint=1)
        v = smax * ((thickness * delta - pos)/(thickness * delta))**m
        return np.r_[v[::-1], v]
    hxy = property(lambda self: self._hxyyzzx.hx)
    hyz = property(lambda self: self._hxyyzzx.hy)
    hzx = property(lambda self: self._hxyyzzx.hz)
    hxz = property(lambda self: self._hxzyxzy.hx)
    hyx = property(lambda self: self._hxzyxzy.hy)
    hzy = property(lambda self: self._hxzyxzy.hz)
    exy = property(lambda self: self._exyyzzx.ex)
    eyz = property(lambda self: self._exyyzzx.ey)
    ezx = property(lambda self: self._exyyzzx.ez)
    exz = property(lambda self: self._exzyxzy.ex)
    eyx = property(lambda self: self._exzyxzy.ey)
    ezy = property(lambda self: self._exzyxzy.ez)
    sgmx = property(lambda self: self._sgmx)
    sgmy = property(lambda self: self._sgmy)
    sgmz = property(lambda self: self._sgmz)
    time = property(lambda self: self._time)
    spacing = property(lambda self: self._spacing)
    dx = property(lambda self: self._spacing[0])
    dy = property(lambda self: self._spacing[1])
    dz = property(lambda self: self._spacing[2])
    dt = property(lambda self: self._time.dt)
    def __init__(self, spacing, time, sigma):
        self._spacing = spacing
        self._time = time
        self._sgmx, self._sgmy, self._sgmz = x, y, z = sigma
        self._hxyyzzx = HField(len(x), len(y), len(z))
        self._hxzyxzy = HField(len(x), len(y), len(z))
        self._exyyzzx = EField(len(x), len(y), len(z))
        self._exzyxzy = EField(len(x), len(y), len(z))
        
class PMLCalc(object):
    pml = property(lambda self: self._pml)
    def __init__(self, pml):
        self._pml = pml
    def calc_const(self, sgm, delta):
        dt = self.pml.dt
        c0 = 2. * dt / delta / (2. * EP0 + sgm * dt)
        c1 = (2. * EP0 - sgm * dt) / (2. * EP0 + sgm * dt)
        return np.array(c0), np.array(c1)
        
class PMLCalcEMN(PMLCalc):
    def __init__(self, pml, sgm, delta, shape):
        super().__init__(pml)
        c0, c1 = self.calc_const(sgm, delta)
        self._const = c0.reshape(*shape), c1.reshape(*shape)
    def calc_const(self, sgm, delta):
        c0, c1 = super().calc_const(.5 * (sgm[:-1] + sgm[1:]), delta)
        return c0, c1
    def calculate(self, e, h0, h1, slices, d0slices, d1slices, sign=1):
        c0, c1 = self._const
        e.__setitem__(slices,
                      sign * 
                      c0 * (h0.__getitem__(d1slices) - h0.__getitem__(d0slices) +
                            h1.__getitem__(d1slices) - h1.__getitem__(d0slices)) +
                      c1 * e.__getitem__(slices)
                     )
    
class PMLCalcHMN(PMLCalc):
    def __init__(self, pml, sgm, delta, shape):
        super().__init__(pml)
        c0, c1 = self.calc_const(sgm, delta)
        self._const = c0.reshape(*shape), c1.reshape(*shape)
    def calc_const(self, sgm, delta):
        c0, c1 = super().calc_const(sgm, delta)
        return EP0 / MU0 * c0, c1
    def calculate(self, h, e0, e1, d0slices, d1slices, sign=1):
        slices = (slice(None), slice(None), slice(None))
        c0, c1 = self._const
        #print(type(self), c1.shape, h.shape)
        h.__setitem__(slices,
                      sign * 
                      c0 * (e0.__getitem__(d1slices) - e0.__getitem__(d0slices) +
                            e1.__getitem__(d1slices) - e1.__getitem__(d0slices)) +
                      c1 * h.__getitem__(slices)
                     )
    
class PMLCalcEXY(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmy, pml.dy, shape=(1, len(pml.sgmy) - 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.exz,
            self._pml.hzx, self._pml.hzy,
            slices=(slice(None), slice(1,-1), slice(1,-1)),
            d0slices=(slice(None), slice(None, -1), slice(1, -1)),
            d1slices=(slice(None), slice(1, None), slice(1, -1)),
        )
        
class PMLCalcEXZ(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmz, pml.dz, shape=(1, 1, len(pml.sgmz) - 1))
    def calculate(self):
        super().calculate(
            self._pml.exz,
            self._pml.hyz, self._pml.hyx,
            slices=(slice(None), slice(1,-1), slice(1,-1)),
            d0slices=(slice(None), slice(1, -1), slice(None, -1)),
            d1slices=(slice(None), slice(1, -1), slice(1, None)),
            sign = -1,
        )
    
class PMLCalcEYZ(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmz, pml.dz, shape=(1, 1, len(pml.sgmz) - 1))
    def calculate(self):
        super().calculate(
            self._pml.eyz,
            self._pml.hxy, self._pml.hxz,
            slices=(slice(1, -1), slice(None), slice(1, -1)),
            d0slices=(slice(1, -1), slice(None), slice(None, -1)),
            d1slices=(slice(1, -1), slice(None), slice(1, None)),
        )
    
class PMLCalcEYX(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmx, pml.dx, shape=(len(pml.sgmx) - 1, 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.eyx,
            self._pml.hzx, self._pml.hzy,
            slices=(slice(1, -1), slice(None), slice(1, -1)),
            d0slices=(slice(None, -1), slice(None), slice(1, -1)),
            d1slices=(slice(1, None), slice(None), slice(1, -1)),
            sign = -1
        )
        
class PMLCalcEZX(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmx, pml.dx, shape=(len(pml.sgmx) - 1, 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.ezx,
            self._pml.hyz, self._pml.hyx,
            slices=(slice(1, -1), slice(1, -1), slice(None)),
            d0slices=(slice(None, -1), slice(1, -1), slice(None)),
            d1slices=(slice(1, None), slice(1, -1), slice(None)),
        )
    
class PMLCalcEZY(PMLCalcEMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmy, pml.dy, shape=(1, len(pml.sgmy) - 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.ezy,
            self._pml.hxy, self._pml.hxz,
            slices=(slice(1, -1), slice(1, -1), slice(None)),
            d0slices=(slice(1, -1), slice(None, -1), slice(None)),
            d1slices=(slice(1, -1), slice(1, None), slice(None)),
            sign = -1
        )
    
class PMLCalcHXY(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmy, pml.dy, shape=(1, len(pml.sgmy), 1))
    def calculate(self):
        super().calculate(
            self._pml.hxy,
            self._pml.ezx, self._pml.ezy,
            d0slices=(slice(None), slice(None, -1), slice(None)),
            d1slices=(slice(None), slice(1, None), slice(None)),
            sign = -1,
        )
    
class PMLCalcHXZ(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmz, pml.dz, shape=(1, 1, len(pml.sgmz)))
    def calculate(self):
        super().calculate(
            self._pml.hxz,
            self._pml.eyz, self._pml.eyx,
            d0slices=(slice(None), slice(None), slice(None, -1)),
            d1slices=(slice(None), slice(None), slice(1, None)),
        )
    
class PMLCalcHYZ(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmz, pml.dz, shape=(1, 1, len(pml.sgmz)))
    def calculate(self):
        super().calculate(
            self._pml.hyz,
            self._pml.exy, self._pml.exz,
            d0slices=(slice(None), slice(None), slice(None, -1)),
            d1slices=(slice(None), slice(None), slice(1, None)),
            sign = -1,
        )
    
class PMLCalcHYX(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmx, pml.dx, shape=(len(pml.sgmx), 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.hyx,
            self._pml.ezx, self._pml.ezy,
            d0slices=(slice(None, -1), slice(None), slice(None)),
            d1slices=(slice(1, None), slice(None), slice(None)),
        )
    
class PMLCalcHZX(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmx, pml.dx, shape=(len(pml.sgmx), 1, 1))
    def calculate(self):
        super().calculate(
            self._pml.hzx,
            self._pml.eyz, self._pml.eyx,
            d0slices=(slice(None, -1), slice(None), slice(None)),
            d1slices=(slice(1, None), slice(None), slice(None)),
            sign = -1,
        )
    
class PMLCalcHZY(PMLCalcHMN):
    def __init__(self, pml):
        super().__init__(pml, pml.sgmy, pml.dy, shape=(1, len(pml.sgmy), 1))
    def calculate(self):
        super().calculate(
            self._pml.hzy,
            self._pml.exy, self._pml.exz,
            d0slices=(slice(None), slice(None, -1), slice(None)),
            d1slices=(slice(None), slice(1, None), slice(None)),
        )

class PMLCalcM(object):
    pml = property(lambda self: self._pml)
    def __init__(self, pml):
        self._pml = pml
        self._calculator = []
    def calculate(self):
        for o in self._calculator:
            o.calculate()

class PMLCalcH(PMLCalcM):
    def __init__(self, pml):
        super().__init__(pml)
        self._calculator = [
            PMLCalcHXY(pml), PMLCalcHXZ(pml),
            PMLCalcHYZ(pml), PMLCalcHYX(pml),
            PMLCalcHZX(pml), PMLCalcHZY(pml),
        ]

class PMLCalcE(PMLCalcM):
    def __init__(self, pml):
        super().__init__(pml)
        self._calculator = [
            PMLCalcEXY(pml), PMLCalcEXZ(pml),
            PMLCalcEYZ(pml), PMLCalcEYX(pml),
            PMLCalcEZX(pml), PMLCalcEZY(pml),
        ]
    
class PMLCommunicator(object):
    x = property(lambda self: self._x)
    y = property(lambda self: self._y)
    z = property(lambda self: self._z)
    xy = property(lambda self: self._xy)
    yz = property(lambda self: self._yz)
    zx = property(lambda self: self._zx)
    xyz = property(lambda self: self._xyz)
    yee = property(lambda self: self._yee)
    def __init__(self, yee=None, x=None, y=None, z=None, xy=None, yz=None, zx=None, xyz=None):
        self._x, self._y, self._z = x, y, z
        self._xy, self._yz, self._zx = xy, yz, zx
        self._xyz = xyz
        self._yee = yee
    def communicate(self):
        pass

class PMLComEX(PMLCommunicator):
    def communicate(self):
        x = self.x
        ex, ey, ez = self.yee.e
        ey[-1,:,:] = x.eyz[ 1,:,:] + x.eyx[ 1,:,:]
        ez[-1,:,:] = x.ezx[ 1,:,:] + x.ezy[ 1,:,:]
        ey[ 0,:,:] = x.eyz[-2,:,:] + x.eyx[-2,:,:]
        ez[ 0,:,:] = x.ezx[-2,:,:] + x.ezy[-2,:,:]
    
class PMLComEY(PMLCommunicator):
    def communicate(self):
        y = self.y
        ex, ey, ez = self.yee.e
        ez[:,-1,:] = y.ezx[:, 1,:] + y.ezy[:, 1,:]
        ex[:,-1,:] = y.exy[:, 1,:] + y.exz[:, 1,:]
        ez[:, 0,:] = y.ezx[:,-2,:] + y.ezy[:,-2,:]
        ex[:, 0,:] = y.exy[:,-2,:] + y.exz[:,-2,:]
        
class PMLComEZ(PMLCommunicator):
    def communicate(self):
        z = self.z
        ex, ey, ez = self.yee.e
        ex[:,:,-1] = z.exy[:,:, 1] + z.exz[:,:, 1]
        ey[:,:,-1] = z.eyz[:,:, 1] + z.eyx[:,:, 1]
        ex[:,:, 0] = z.exy[:,:,-2] + z.exz[:,:,-2]
        ey[:,:, 0] = z.eyz[:,:,-2] + z.eyx[:,:,-2]
        
class PMLComEXY(PMLCommunicator):
    def communicate(self):
        x, y, xy = self.x, self.y, self.xy
        x.ezx[:,-1,:], x.ezx[:, 0,:] = xy.ezx[:, 1,:], xy.ezx[:,-2,:]
        x.ezy[:,-1,:], x.ezy[:, 0,:] = xy.ezy[:, 1,:], xy.ezy[:,-2,:]
        x.exy[:,-1,:], x.exy[:, 0,:] = xy.exy[:, 1,:], xy.exy[:,-2,:]
        x.exz[:,-1,:], x.exz[:, 0,:] = xy.exz[:, 1,:], xy.exz[:,-2,:]
        y.eyz[-1,:,:], y.eyz[ 0,:,:] = xy.eyz[ 1,:,:], xy.eyz[-2,:,:]
        y.eyx[-1,:,:], y.eyx[ 0,:,:] = xy.eyx[ 1,:,:], xy.eyx[-2,:,:]
        y.ezx[-1,:,:], y.ezx[ 0,:,:] = xy.ezx[ 1,:,:], xy.ezx[-2,:,:]
        y.ezy[-1,:,:], y.ezy[ 0,:,:] = xy.ezy[ 1,:,:], xy.ezy[-2,:,:]
    
class PMLComEYZ(PMLCommunicator):
    def communicate(self):
        y, z, yz = self.y, self.z, self.yz
        y.exy[:,:,-1], y.exy[:,:, 0] = yz.exy[:,:, 1], yz.exy[:,:,-2]
        y.exz[:,:,-1], y.exz[:,:, 0] = yz.exz[:,:, 1], yz.exz[:,:,-2]
        y.eyz[:,:,-1], y.eyz[:,:, 0] = yz.eyz[:,:, 1], yz.eyz[:,:,-2]
        y.eyx[:,:,-1], y.eyx[:,:, 0] = yz.eyx[:,:, 1], yz.eyx[:,:,-2]
        z.ezx[:,-1,:], z.ezx[:, 0,:] = yz.ezx[:, 1,:], yz.ezx[:,-2,:]
        z.ezy[:,-1,:], z.ezy[:, 0,:] = yz.ezy[:, 1,:], yz.ezy[:,-2,:]
        z.exy[:,-1,:], z.exy[:, 0,:] = yz.exy[:, 1,:], yz.exy[:,-2,:]
        z.exz[:,-1,:], z.exz[:, 0,:] = yz.exz[:, 1,:], yz.exz[:,-2,:]
        
class PMLComEZX(PMLCommunicator):
    def communicate(self):
        z, x, zx = self.z, self.x, self.zx
        z.eyz[-1,:,:], z.eyz[ 0,:,:] = zx.eyz[ 1,:,:], zx.eyz[-2,:,:]
        z.eyx[-1,:,:], z.eyx[ 0,:,:] = zx.eyx[ 1,:,:], zx.eyx[-2,:,:]
        z.ezx[-1,:,:], z.ezx[ 0,:,:] = zx.ezx[ 1,:,:], zx.ezx[-2,:,:]
        z.ezy[-1,:,:], z.ezy[ 0,:,:] = zx.ezy[ 1,:,:], zx.ezy[-2,:,:]
        x.exy[:,:,-1], x.exy[:,:, 0] = zx.exy[:,:, 1], zx.exy[:,:,-2]
        x.exz[:,:,-1], x.exz[:,:, 0] = zx.exz[:,:, 1], zx.exz[:,:,-2]
        x.eyz[:,:,-1], x.eyz[:,:, 0] = zx.eyz[:,:, 1], zx.eyz[:,:,-2]
        x.eyx[:,:,-1], x.eyx[:,:, 0] = zx.eyx[:,:, 1], zx.eyx[:,:,-2]
        
class PMLComEXYZ(PMLCommunicator):
    def communicate(self):
        xy, yz, zx, xyz = self.xy, self.yz, self.zx, self.xyz
        yz.eyz[-1,:,:], yz.eyz[ 0,:,:] = xyz.eyz[ 1,:,:], xyz.eyz[-2,:,:]
        yz.eyx[-1,:,:], yz.eyx[ 0,:,:] = xyz.eyx[ 1,:,:], xyz.eyx[-2,:,:]
        yz.ezx[-1,:,:], yz.ezx[ 0,:,:] = xyz.ezx[ 1,:,:], xyz.ezx[-2,:,:]
        yz.ezy[-1,:,:], yz.ezy[ 0,:,:] = xyz.ezy[ 1,:,:], xyz.ezy[-2,:,:]
        zx.ezx[:,-1,:], zx.ezx[:, 0,:] = xyz.ezx[:, 1,:], xyz.ezx[:,-2,:]
        zx.ezy[:,-1,:], zx.ezy[:, 0,:] = xyz.ezy[:, 1,:], xyz.ezy[:,-2,:]
        zx.exy[:,-1,:], zx.exy[:, 0,:] = xyz.exy[:, 1,:], xyz.exy[:,-2,:]
        zx.exz[:,-1,:], zx.exz[:, 0,:] = xyz.exz[:, 1,:], xyz.exz[:,-2,:]
        xy.exy[:,:,-1], xy.exy[:,:, 0] = xyz.exy[:,:, 1], xyz.exy[:,:,-2]
        xy.exz[:,:,-1], xy.exz[:,:, 0] = xyz.exz[:,:, 1], xyz.exz[:,:,-2]
        xy.eyz[:,:,-1], xy.eyz[:,:, 0] = xyz.eyz[:,:, 1], xyz.eyz[:,:,-2]
        xy.eyx[:,:,-1], xy.eyx[:,:, 0] = xyz.eyx[:,:, 1], xyz.eyx[:,:,-2]
        
class PMLComHX(PMLCommunicator):
    def communicate(self):
        x = self.x
        hx, hy, hz = self.yee.h
        x.hyz[ 0,:,:], x.hyx[ 0,:,:] = hy[-1,:,:], 0.
        x.hzx[ 0,:,:], x.hzy[ 0,:,:] = hz[-1,:,:], 0.
        x.hyz[-1,:,:], x.hyx[-1,:,:] = hy[ 0,:,:], 0.
        x.hzx[-1,:,:], x.hzy[-1,:,:] = hz[ 0,:,:], 0.
    
class PMLComHY(PMLCommunicator):
    def communicate(self):
        y = self.y
        hx, hy, hz = self.yee.h
        y.hxy[:, 0,:], y.hxz[:, 0,:] = hx[:,-1,:], 0.
        y.hzx[:, 0,:], y.hzy[:, 0,:] = hz[:,-1,:], 0.
        y.hxy[:,-1,:], y.hxz[:,-1,:] = hx[:, 0,:], 0.
        y.hzx[:,-1,:], y.hzy[:,-1,:] = hz[:, 0,:], 0.
        
class PMLComHZ(PMLCommunicator):
    def communicate(self):
        z = self.z
        hx, hy, hz = self.yee.h
        z.hxy[:,:, 0], z.hxz[:,:, 0] = hx[:,:,-1], 0.
        z.hyz[:,:, 0], z.hyx[:,:, 0] = hy[:,:,-1], 0.
        z.hxy[:,:,-1], z.hxz[:,:,-1] = hx[:,:, 0], 0.
        z.hyz[:,:,-1], z.hyx[:,:,-1] = hy[:,:, 0], 0.
        
class PMLComHXY(PMLCommunicator):
    def communicate(self):
        x, y, xy = self.x, self.y, self.xy
        xy.hxy[:, 0,:], xy.hxy[:,-1,:] = x.hxy[:,-1,:], x.hxy[:, 0,:]
        xy.hxz[:, 0,:], xy.hxz[:,-1,:] = x.hxz[:,-1,:], x.hxz[:, 0,:]
        xy.hzx[:, 0,:], xy.hzx[:,-1,:] = x.hzx[:,-1,:], x.hzx[:, 0,:]
        xy.hzy[:, 0,:], xy.hzy[:,-1,:] = x.hzy[:,-1,:], x.hzy[:, 0,:]
        xy.hyz[ 0,:,:], xy.hyz[-1,:,:] = y.hyz[-1,:,:], y.hyz[ 0,:,:]
        xy.hyx[ 0,:,:], xy.hyx[-1,:,:] = y.hyx[-1,:,:], y.hyx[ 0,:,:]
        xy.hzx[ 0,:,:], xy.hzx[-1,:,:] = y.hzx[-1,:,:], y.hzx[ 0,:,:]
        xy.hzy[ 0,:,:], xy.hzy[-1,:,:] = y.hzy[-1,:,:], y.hzy[ 0,:,:]
        
class PMLComHYZ(PMLCommunicator):
    def communicate(self):
        y, z, yz = self.y, self.z, self.yz
        yz.hxy[:,:, 0], yz.hxy[:,:,-1] = y.hxy[:,:,-1], y.hxy[:,:, 0]
        yz.hxz[:,:, 0], yz.hxz[:,:,-1] = y.hxz[:,:,-1], y.hxz[:,:, 0]
        yz.hyz[:,:, 0], yz.hyz[:,:,-1] = y.hyz[:,:,-1], y.hyz[:,:, 0]
        yz.hyx[:,:, 0], yz.hyx[:,:,-1] = y.hyx[:,:,-1], y.hyx[:,:, 0]
        yz.hxy[:, 0,:], yz.hxy[:,-1,:] = z.hxy[:,-1,:], z.hxy[:, 0,:]
        yz.hxz[:, 0,:], yz.hxz[:,-1,:] = z.hxz[:,-1,:], z.hxz[:, 0,:]
        yz.hzx[:, 0,:], yz.hzx[:,-1,:] = z.hzx[:,-1,:], z.hzx[:, 0,:]
        yz.hzy[:, 0,:], yz.hzy[:,-1,:] = z.hzy[:,-1,:], z.hzy[:, 0,:]
        
class PMLComHZX(PMLCommunicator):
    def communicate(self):
        z, x, zx = self.z, self.x, self.zx
        zx.hyz[ 0,:,:], zx.hyz[-1,:,:] = z.hyz[-1,:,:], z.hyz[ 0,:,:]
        zx.hyx[ 0,:,:], zx.hyx[-1,:,:] = z.hyx[-1,:,:], z.hyx[ 0,:,:]
        zx.hzx[ 0,:,:], zx.hzx[-1,:,:] = z.hzx[-1,:,:], z.hzx[ 0,:,:]
        zx.hzy[ 0,:,:], zx.hzy[-1,:,:] = z.hzy[-1,:,:], z.hzy[ 0,:,:]
        zx.hxy[:,:, 0], zx.hxy[:,:,-1] = x.hxy[:,:,-1], x.hxy[:,:, 0]
        zx.hxz[:,:, 0], zx.hxz[:,:,-1] = x.hxz[:,:,-1], x.hxz[:,:, 0]
        zx.hyz[:,:, 0], zx.hyz[:,:,-1] = x.hyz[:,:,-1], x.hyz[:,:, 0]
        zx.hyx[:,:, 0], zx.hyx[:,:,-1] = x.hyx[:,:,-1], x.hyx[:,:, 0]
        
class PMLComHXYZ(PMLCommunicator):
    def communicate(self):
        xy, yz, zx, xyz = self.xy, self.yz, self.zx, self.xyz
        xyz.hyz[ 0,:,:], xyz.hyz[-1,:,:] = yz.hyz[-1,:,:], yz.hyz[ 0,:,:]
        xyz.hyx[ 0,:,:], xyz.hyx[-1,:,:] = yz.hyx[-1,:,:], yz.hyx[ 0,:,:]
        xyz.hzx[ 0,:,:], xyz.hzx[-1,:,:] = yz.hzx[-1,:,:], yz.hzx[ 0,:,:]
        xyz.hzy[ 0,:,:], xyz.hzy[-1,:,:] = yz.hzy[-1,:,:], yz.hzy[ 0,:,:]
        xyz.hxy[:, 0,:], xyz.hxy[:,-1,:] = zx.hxy[:,-1,:], zx.hxy[:, 0,:]
        xyz.hxz[:, 0,:], xyz.hxz[:,-1,:] = zx.hxz[:,-1,:], zx.hxz[:, 0,:]
        xyz.hzx[:, 0,:], xyz.hzx[:,-1,:] = zx.hzx[:,-1,:], zx.hzx[:, 0,:]
        xyz.hzy[:, 0,:], xyz.hzy[:,-1,:] = zx.hzy[:,-1,:], zx.hzy[:, 0,:]
        xyz.hxy[:,:, 0], xyz.hxy[:,:,-1] = xy.hxy[:,:,-1], xy.hxy[:,:, 0]
        xyz.hxz[:,:, 0], xyz.hxz[:,:,-1] = xy.hxz[:,:,-1], xy.hxz[:,:, 0]
        xyz.hyz[:,:, 0], xyz.hyz[:,:,-1] = xy.hyz[:,:,-1], xy.hyz[:,:, 0]
        xyz.hyx[:,:, 0], xyz.hyx[:,:,-1] = xy.hyx[:,:,-1], xy.hyx[:,:, 0]
        
class Dispatcher(object):
    def __init__(self, methods):
        self._methods = methods
    def dispatch(self):
        for m in self._methods:
            m()
    
class PMLSolver(object):
    def __init__(self, yee, time, **kw):
        g = yee.grid
        x = PML(g.spacing, time, (PML.sigma(g.dx, **kw), np.zeros(g.ny), np.zeros(g.nz)))
        y = PML(g.spacing, time, (np.zeros(g.nx), PML.sigma(g.dy, **kw), np.zeros(g.nz)))
        z = PML(g.spacing, time, (np.zeros(g.nx), np.zeros(g.ny), PML.sigma(g.dz, **kw)))
        xy = PML(g.spacing, time, (PML.sigma(g.dx, **kw), PML.sigma(g.dy, **kw), np.zeros(g.nz)))
        yz = PML(g.spacing, time, (np.zeros(g.nx), PML.sigma(g.dy, **kw), PML.sigma(g.dz, **kw)))
        zx = PML(g.spacing, time, (PML.sigma(g.dx, **kw), np.zeros(g.ny), PML.sigma(g.dz, **kw)))
        xyz = PML(g.spacing, time, (PML.sigma(g.dx, **kw), PML.sigma(g.dy, **kw), PML.sigma(g.dz, **kw)))
        self._boundary = {
            'x': [x,
                  PMLCalcH(x), PMLComHX(yee=yee, x=x),
                  PMLCalcE(x), PMLComEX(yee=yee, x=x)],
            'y': [y,
                  PMLCalcH(y), PMLComHY(yee=yee, y=y),
                  PMLCalcE(y), PMLComEY(yee=yee, y=y)],
            'z': [z,
                  PMLCalcH(z), PMLComHZ(yee=yee, z=z),
                  PMLCalcE(z), PMLComEZ(yee=yee, z=z)],
            'xy': [xy,
                   PMLCalcH(xy), PMLComHXY(x=x, y=y, xy=xy),
                   PMLCalcE(xy), PMLComEXY(x=x, y=y, xy=xy)],
            'yz': [yz,
                   PMLCalcH(yz), PMLComHYZ(y=y, z=z, yz=yz),
                   PMLCalcE(yz), PMLComEYZ(y=y, z=z, yz=yz)],
            'zx': [zx,
                   PMLCalcH(zx), PMLComHZX(z=z, x=x, zx=zx),
                   PMLCalcE(zx), PMLComEZX(z=z, x=x, zx=zx)],
            'xyz': [xyz,
                    PMLCalcH(xyz), PMLComHXYZ(xy=xy, yz=yz, zx=zx, xyz=xyz),
                    PMLCalcE(xyz), PMLComEXYZ(xy=xy, yz=yz, zx=zx, xyz=xyz)],
        }
        self.h_calc = Dispatcher([self._boundary[k][1].calculate for k in self._boundary])
        self.h_com = Dispatcher([self._boundary[k][2].communicate for k in self._boundary])
        self.e_calc = Dispatcher([self._boundary[k][3].calculate for k in self._boundary])
        self.e_com = Dispatcher([self._boundary[k][4].communicate for k in self._boundary])

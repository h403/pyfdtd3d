import base

class Grid(base.Grid):
    def __init__(self):
        base.Grid.__init__(self)
        self._dimensions = None, None, None
        self._spacing = None, None, None
        self._origin = None, None, None
        self.events = {
            'set_dimensions': base.EventHandler(),
            'set_spacing': base.EventHandler(),
            'set_origin': base.EventHandler(),
        }
    def _set_dimensions(self, v): # v: i, j, k
        self._dimensions = v
        self.events['set_dimensions'].dispatch()
    dimensions = property(lambda self: self._dimensions, _set_dimensions)
    def _set_spacing(self, v): # v: di, dj, dk
        self._spacing = v
        self.events['set_spacing'].dispatch()
    spacing = property(lambda self: self._spacing, _set_spacing)
    def _set_origin(self, v): # v: oi, oj, ok
        self._origin = v
        self.events['set_origin'].dispatch()
    origin = property(lambda self: self._origin, _set_origin)

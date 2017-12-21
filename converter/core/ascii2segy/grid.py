# -*- coding: utf-8 -*-
import numpy as np


class Grid(object):
    ''' init a seismic grid then use to convert between x,y and inline xline'''

    def __init__(self, p0, irot, ibin, xbin):
        '''
        p0, is a point in the array with attributes:
            p0.x, p0.y, p0.inline, p0.xline
        irot, is the inline rotation in radians
        ibin is the inline spacing
        xbin is the xline spacing
        '''
        self.irot = irot
        self.ibin = ibin
        self.xbin = xbin
        self.inline = p0.inline
        self.xline = p0.xline
        self.x = p0.x
        self.y = p0.y

    def _sum(self, num, func, inline, xline):
        part1 = self.xbin * (xline - self.xline) * np.sin(self.irot)
        part2 = self.ibin * (inline - self.inline) * func(self.irot)
        return num + part1 + part2

    def ix2xy(self, inline, xline):
        x = self._sum(self.x, np.cos, inline, xline)
        y = self._sum(self.y, np.sin, inline, xline)
        return x, y

    def xy2ix(self, x, y):
        raise ("Not implemented")

#!/usr/bin/env python

"""
Try simulating the PFS spectrograph
"""

import numpy as np
from astropy.io import fits

from specter.psf import PSF
from specter.util.traceset import fit_traces

class PFS_PSF(PSF):
    def __init__(self, filename):
        """
        Initialize PSF with text file that has columns
        
            ifiber wavelength x y
            
        ifiber and wavelength must be monotonically increasing.
        Every fiber must have the same number of wavelength samples, though
        they don't have to be the same wavelengths.
        """
        self.npix_x = 4096
        self.npix_y = 4096

        #- Read data
        ### fiber, wave, x, y = np.loadtxt(filename, unpack=True)
        fiber, wave, y, x = np.loadtxt(filename, unpack=True)
        self.nspec = len(set(fiber))
        self.nwave = len(wave) / self.nspec
        wave = wave[0:self.nwave]
        
        #- make fiber number go in positive x direction
        x = x.reshape( [self.nspec, self.nwave] )
        x = x[-1::-1, :]
        
        #- x and y vs. wavelength arrays
        self._x = fit_traces(wave, x.reshape( [self.nspec, self.nwave] ))
        self._y = fit_traces(wave, y.reshape( [self.nspec, self.nwave] ))
        self._xmin = np.min(x)
        self._xmax = np.max(x)
        self._ymin = np.min(y)
        self._ymax = np.max(y)
        ### self._w = self._x.invert()
        self._w = self._y.invert()
        self._wmin = np.min(wave)
        self._wmax = np.max(wave)

        #- Treat sigma as constant for now, but put into arrays for
        #- interpolation for later updates
        self._sigx = fit_traces(wave, np.ones([self.nspec, self.nwave]) * 1.2)
        self._sigy = fit_traces(wave, np.ones([self.nspec, self.nwave]) * 1.2)
        
    # def x(self, ispec, wavelength):
    #     return np.interp(wavelength, self._wave[ispec], self._x[ispec])
    # 
    # def y(self, ispec, wavelength):
    #     return np.interp(wavelength, self._wave[ispec], self._y[ispec])
    #     

    #- PFS spectra have wavelength in the x direction, but some specter code
    #- assumes that wavelength goes in the y direction.  When calling
    #- wavelength with either x or y, treat it as x
    # def wavelength(self, ispec, x=None, y=None):
    #     if y is not None:
    #         return self._w.eval(ispec, y)
    #     elif x is not None:
    #         return self._w.eval(ispec, x)
    #     else:
    #         return self._w.eval(ispec, np.arange(self.npix_x))
                
    def _xypix(self, ispec, wavelength):
        xc = self.x(ispec, wavelength)
        yc = self.y(ispec, wavelength)
        sigx = self._sigx.eval(ispec, wavelength)
        sigy = self._sigy.eval(ispec, wavelength)
        
        xc0 = xc - int(xc)
        yc0 = yc - int(yc)
        
        d = 5
        x = np.arange(-d, +d+1)
        xx, yy = np.meshgrid(x, x)
        image = np.exp(-(xx-xc0)**2 / (2*sigx**2)) * np.exp(-(yy-yc0)**2 / (2*sigy**2))
        image /= np.sum(image)
        
        xslice = slice(int(xc)-d, int(xc)+d+1)
        yslice = slice(int(yc)-d, int(yc)+d+1)
        
        return xslice, yslice, image
    
    
        


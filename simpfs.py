#!/usr/bin/env python

"""
Try simulating the PFS spectrograph
"""

import numpy as np
from astropy.io import fits

from specter.psf import PSF

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
        fiber, wave, x, y = np.loadtxt(filename, unpack=True)
        self.nspec = len(set(fiber))
        self.nwave = len(wave) / self.nspec
        
        #- x and y vs. wavelength arrays
        self._x = x.reshape( [self.nspec, self.nwave] )
        self._y = y.reshape( [self.nspec, self.nwave] )
        self._wave = wave.reshape( [self.nspec, self.nwave] )
        self._wmin = np.min(self._wave)
        self._wmax = np.max(self._wave)

        #- Treat sigma as constant for now, but put into arrays for
        #- interpolation for later updates
        self._sigx = np.ones( [self.nspec, self.nwave] ) * 1.2
        self._sigy = np.ones( [self.nspec, self.nwave] ) * 1.2
        
    def x(self, ispec, wavelength):
        return np.interp(wavelength, self._wave[ispec], self._x[ispec])

    def y(self, ispec, wavelength):
        return np.interp(wavelength, self._wave[ispec], self._y[ispec])
        
    def wavelength(self, ispec, x=None, y=None):
        if y is not None:
            x = y
            return np.interp(x, self._x[ispec], self._wave[ispec])
        else:
            return np.interp(x, self._x[ispec], self._wave[ispec])
        
    def _xypix(self, ispec, wavelength):
        xc = self.x(ispec, wavelength)
        yc = self.y(ispec, wavelength)
        sigx = np.interp(wavelength, self._wave[ispec], self._sigx[ispec])
        sigy = np.interp(wavelength, self._wave[ispec], self._sigy[ispec])
        
        xc0 = xc - int(xc)
        yc0 = yc - int(yc)
        
        d = 10
        x = np.arange(-d, +d+1)
        xx, yy = np.meshgrid(x, x)
        image = np.exp(-(xx-xc0)**2 / (2*sigx**2)) * np.exp(-(yy-yc0)**2 / (2*sigy**2))
        image /= np.sum(image)
        
        xslice = slice(int(xc)-d, int(xc)+d+1)
        yslice = slice(int(yc)-d, int(yc)+d+1)
        
        return xslice, yslice, image
    
    
        


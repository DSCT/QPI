# -*- coding: utf-8 -*-

# This script implements a DCT-based one-step path independent phase
# unwrapping method described in DOI: 10.3390/jimaging1010031 .
# Phase will be unwrapped up to some additive constant.
# - Unwrapping performance depends on the phase map's noise level and C2 smoothness.
# - If zeropadding has to be applied beforehand for FFTs to work, some ugly
#   boundary effects might occur. Best use dims supported by FFTs.
# - Masking out parts of the phase map with zeros also leads to boundary effects.
# - Use it stand alone or as initial guess for preconditioned conjugate gradient method.

from __future__ import division
import numpy as np, arrayfire as af, matplotlib.pyplot as pl
from time import time

# 1D DCT, results are the same as scipy.fftpack.dct()
def dct1(arr, norm=None):
	N = arr.dims()[0]
	out = 2 * af.real( af.exp(-0.5j*np.pi/N*af.range(*arr.dims(),dim=0,dtype=arr.dtype())) * af.fft(af.join(0,arr[0:N:2],af.flip(arr[1:N:2]))) )
	if norm=='ortho':
		out /= np.sqrt(2*N)
		out[0] /= np.sqrt(2)
	return out

# 1D IDCT, results are the same as scipy.fftpack.idct()
def idct1(arr, norm=None):
	N = arr.dims()[0]
	tmp = arr.copy()
	offset = af.tile(tmp[0],N)
	tmp[0] = 0.
	tmp = 2 * af.real( af.ifft( af.exp(0.5j*np.pi/N*af.range(*arr.dims(),dim=0,dtype=arr.dtype())) * tmp ) * N )
	out = af.constant(0,*arr.dims(),dtype=arr.dtype())
	out[0:N:2] = tmp[:N//2]
	out[1:N:2] = af.flip(tmp[N//2:])
	if norm=='ortho':
		offset /= np.sqrt(N)
		out /= np.sqrt(2*N)
	out += offset
	tmp = offset = None
	return out

# 2D DCT and inverse
def dct2(arr, inverse=False):
	if inverse:
		return af.transpose(idct1(af.transpose(idct1(arr,norm='ortho')),norm='ortho'))
	else:
		return af.transpose(dct1(af.transpose(dct1(arr,norm='ortho')),norm='ortho'))

# Laplace operator and inverse
def laplacian(arr, inverse=False):
	if inverse:
		return dct2(dct2(arr)/coord, inverse=True) # factor -M*N/4/pi**2 omitted
	else:
		return dct2(dct2(arr)*coord, inverse=True) # factor -4*pi**2/M/N omitted

# phase unwrapping, see DOI: 10.3390/jimaging1010031
def unwrap_dct(arr):
	return arr + af.round( ( laplacian( af.cos(arr)*laplacian(af.sin(arr)) - af.sin(arr)*laplacian(af.cos(arr)), inverse=True ) - arr ) / 2/np.pi ) * 2*np.pi

# phase map dims, zeropadding and masking should be avoided if possible
N = 1024
M = 1024

# make some coordinates for the synthetic phase map
x = af.range(N,M,dim=1)
y = af.range(N,M,dim=0)
x /= af.max(x)/np.pi/2
y /= af.max(y)/np.pi/2

snr = 10 # SNR for the Gaussian noise

# some synthetic phase maps
phase_orig = af.sin(x)*x+af.cos(y)*y
#phase_orig = af.sin(x*y)*x+af.cos(x*x+y*y)*y
#phase_orig = (x-af.max(x)/2)**2 + (y-af.max(y)/2)**2
x = y = None
phase_orig /= af.max(af.abs(phase_orig)) / np.pi/4 # scale phase map amplitude
phase_orig += af.to_array( np.random.normal(scale=np.sqrt(0.5/snr),size=(N,M)).astype(np.float32) ) # add noise

phase_wrapped = af.atan2( af.sin(phase_orig), af.cos(phase_orig) ) # wrap phase map to [-pi,+pi]

# precompute Fourier domain coordinates for the Laplacians of phase_wrapped
m = af.range(N,M,dim=1)
n = af.range(N,M,dim=0)
m[:,0] = 1/np.sqrt(M)
n[0,:] = 1/np.sqrt(N)
coord = m*m+n*n
m = n = None

k = 1 # timing iterations

phase_unwrapped = unwrap_dct(phase_wrapped) # warm-up
af.device.sync()
t0=time()
for i in range(k):
	phase_unwrapped = unwrap_dct(phase_wrapped)
af.device.sync()
t1 = time()-t0
af.device.device_gc()
coord = None
print( 'time:\t%fs'%(t1/k))

print( 'norm:\t%f'%af.norm(phase_unwrapped-phase_orig) )

fig, ((ax1,ax2),(ax3,ax4)) = pl.subplots(2,2)
im1 = ax1.imshow(phase_orig.__array__())
pl.colorbar(im1,ax=ax1)
ax1.set_title('original phase, SNR=%i'%snr)
im2 = ax2.imshow(phase_wrapped.__array__())
pl.colorbar(im2,ax=ax2)
ax2.set_title('wrapped phase')
im3 = ax3.imshow(phase_unwrapped.__array__())
pl.colorbar(im3,ax=ax3)
ax3.set_title('unwrapped phase')
im4 = ax4.imshow((phase_unwrapped-phase_orig).__array__())
pl.colorbar(im4,ax=ax4)
ax4.set_title('difference')
pl.show()

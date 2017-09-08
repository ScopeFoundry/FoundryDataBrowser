'''
Created on Jun 23, 2017

@author: dbdurham
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from skimage.feature.register_translation import _upsampled_dft, _compute_error, _compute_phasediff

def register_translation_hybrid(src_image, target_image, exponent = 1, upsample_factor=1,
                         space="real"):
    """
    Efficient subpixel image translation registration by hybrid-correlation (cross and phase).
    Exponent = 1 -> cross correlation, exponent = 0 -> phase correlation.
    Closer to zero is more precise but more susceptible to noise.
    
    This code gives the same precision as the FFT upsampled correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    exponent: float, optional
        Power to which amplitude contribution to correlation is raised.
        exponent = 0: Phase correlation
        exponent = 1: Cross correlation
        0 < exponent < 1 = Hybrid
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier"
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image = np.array(src_image, dtype=np.complex128, copy=False)
        target_image = np.array(target_image, dtype=np.complex128, copy=False)
        src_freq = np.fft.fftn(src_image)
        target_freq = np.fft.fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute hybrid-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    amplitude = np.abs(image_product)
    phase = np.angle(image_product)
    total_fourier = amplitude**exponent * np.exp(phase * 1j)
    correlation = np.fft.ifftn(total_fourier)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(correlation)),
                              correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        correlation = _upsampled_dft(total_fourier.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.abs(correlation)),
                              correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        CCmax = correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, _compute_error(CCmax, src_amp, target_amp),\
        _compute_phasediff(CCmax)
    
def shift_subpixel(I, dx=0, dy=0):
    # Shift an image with subpixel precision using the Fourier shift theorem
    
    # Image to shift
    G = np.fft.fft2(I)

    # Prepare inverse pixel coordinate arrays

    qy = np.array([np.fft.fftfreq(G.shape[0]),])
    qy = np.repeat(qy.T, G.shape[1], axis=1)

    qx = np.array([np.fft.fftfreq(G.shape[1]),])
    qx = np.repeat(qx, G.shape[0], axis=0)

    # Calculate shift plane wave
    array_exp = np.exp(-2*np.pi*1j*(qx*dx + qy*dy))

    # Perform shift
    I_shift = np.fft.ifft2(G*array_exp)
    
    return I_shift

def compute_pairwise_shifts(imstack):
    # Calculates the pairwise shifts for images in a stack of format [frame, x, y].
    # returns shift vector as [y, x] for each pair, a 2 x N-1 array where N is num_frames
    
    scan_shape = imstack.shape
    num_pairs = scan_shape[0]-1 
    print('Correcting ' + str(num_pairs) + ' frames...')

    # Prepare window function (Hann)
    win = np.outer(np.hanning(scan_shape[1]),np.hanning(scan_shape[2]))

    # Pairwise shifts
    shift = np.zeros((2, num_pairs))
    for iPair in range(0, num_pairs):
        image = imstack[iPair]
        offset_image = imstack[iPair+1]
        shift[:,iPair], error, diffphase = register_translation_hybrid(image*win, offset_image*win, 
                                                                        exponent = 0.3, upsample_factor = 100)
        # Shifts are defined as [y, x] where y is shift of imaging location 
        # with respect to positive y axis, similarly for x
    return shift

def compute_retained_box(shift_cumul, imshape):
    # Computes coordinates and dimensions of area of image in view throughout the entire stack
    # Uses cumulative shift vector [y, x] for each image, a 2 x N array with N = num_frames
    # imshape is a tuple containing the (y, x) dimensions of the image in pixels

    shift_cumul_y = shift_cumul[0,:]
    shift_cumul_x = shift_cumul[1,:]

    # NOTE: scan_shape indices 2, 3 correspond to y, x
    y1 = int(round(np.max(shift_cumul_y[shift_cumul_y >= 0])+0.001, 0))
    y2 = int(round(imshape[0] + np.min(shift_cumul_y[shift_cumul_y <= 0])-0.001, 0))
    x1 = int(round(np.max(shift_cumul_x[shift_cumul_x >= 0])+0.001, 0))
    x2 = int(round(imshape[1] + np.min(shift_cumul_x[shift_cumul_x <= 0])-0.001, 0))

    boxfd = np.array([y1,y2,x1,x2])
    boxdims = (boxfd[1]-boxfd[0], boxfd[3]-boxfd[2])
    return boxfd, boxdims

def align_image_stack(imstack, shift_cumul_set, boxfd):    
    # Use fourier shift theorem to shift and align the images
    # Takes imageset of format [frames, x, y]
    # shift_cumul_set is the cumulative shifts associated with the image stack
    # boxfd is the coordinate array [y1,y2,x1,x2] of the region in the original image that 
    # is conserved throughout the image stack
    
    for iFrame in range(0, num_frames):
        imstack[iFrame,:,:] = shift_subpixel(imstack[iFrame,:,:], 
                                             dx=shift_cumul_set[1, iFrame], 
                                             dy=shift_cumul_set[0, iFrame])
    # Keep only preserved data
    imstack = np.real(imstack[:,boxfd[0]:boxfd[1], boxfd[2]:boxfd[3]])
    return imstack
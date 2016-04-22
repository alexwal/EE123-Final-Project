from __future__ import division
import numpy as np
import pywt
from wavelet import *

# Compression functions

def compress_bw(im, fraction_coeffs, step_size, wvlt='db4', level=1):
    # step_size: quantization step size
    # DC Level Shift
    # Compute dwt2
    # Threshhold wavelet coefficients
    # Quantize coeffs (these can be transmitted to reconsxt image later)

    im, bit_depth = subtract_bit_depth(im)
    dwt = dwt2(im, level, wvlt)
    if fraction_coeffs != 1:
        thresh = thresh_dwt(dwt, f=fraction_coeffs)
    thresh = scalar_quantizer(thresh, step_size)
    return np.int8(thresh), bit_depth
    
def decompress_bw(threshed_dwt, original_shape, original_bit_depth, step_size, delta=0.5, wvlt='db4', level=1):
    # inverse of the above compression function.
    # if original_bit_depth = 8 (ie 0-255 values), then add 2**7 because we sub'd 2**7 before.
    threshed_dwt = scalar_dequantizer(threshed_dwt, step_size=step_size, delta=delta)
    compressed_img = idwt2(threshed_dwt, wvlt=wvlt, levels=level) + 2**(original_bit_depth-1)
    return compressed_img

# Utility functions

def subtract_bit_depth(image):
    # Level shift: (helps with getting chains of zeros in DWT)
    bit_depth = np.log(np.abs(image).max()-1)/np.log(2)//1 + 1
    image = image - 2**(bit_depth-1)
    return image, bit_depth

def bit_depth_to_add(image):
	# returns the bit depth to be added back to the uncompressed image.
    bit_depth = np.log(image.max()-1)/np.log(2)//1
    return bit_depth

def get_multiple_shape(shape, block_size):
    # finds the shape where each dimension is a multiple of block_size
    horiz_blocks = shape[0]//block_size
    if horiz_blocks*block_size != shape[0]:
        # we add a block to this dim so blocks fit w/out overlap
        horiz_blocks += 1

    vert_blocks = shape[1]//block_size
    if vert_blocks*block_size != shape[1]:
        vert_blocks += 1

    new_shape = (horiz_blocks*block_size, vert_blocks*block_size)
    return new_shape

def map2multiple(image, block_size):
    # map the original image so dims are a multiple of block_size.
    horiz, vert = get_multiple_shape(image.shape, block_size)
    new_image = np.zeros((horiz, vert))
    new_image[:image.shape[0],:image.shape[1]] = image
    return new_image

def map2original(image, shape):
    # map back to the original image.
    # inverse of map_overlap...
    original = np.zeros(shape)
    original = image[:shape[0],:shape[1]]
    return original

def image2blocks(image, block_size):
    # returns list of blocks
    # scans left to right, top to bot.
    blocks = []
    for i in range(int(image.shape[0]/block_size)):
        for j in range(int(image.shape[1]/block_size)):
            blocks += [image[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]]
    return np.array(blocks)

def blocks2image(blocks, blocks_per_row, block_size):
    # Calculate blocks per column
    # Untile blocks
    # Reshape blocks
    blocks_per_column = int(blocks.shape[0]/blocks_per_row + 1*((blocks.shape[0]%blocks_per_row)!=0))
    blocks = blocks.reshape(blocks.shape[0],block_size,block_size)
    tiled = []
    for i in range(0,blocks_per_column*blocks_per_row,blocks_per_row):
        tiled.append(np.hstack(blocks[i:i+blocks_per_row,:,:]))

    return np.vstack(tiled)

def thresh_dwt(dwt, f):
    # f: the fraction f largest wavelet coeffs (to save)
    # parts borrowed from hw9
    m = np.sort(abs(dwt.ravel()))[::-1]
    idx = int(len(m) * f) # the fraction f largest wavelet coeff.
    thr = m[idx] # threshhold
    return dwt * (abs(dwt) > thr)

def max_decomp_level(shape, wvlt='bior4.4'):
    # returns the maximum useful level of decomposition for the given input data shape and wavelet.
    wv = pywt.Wavelet(wvlt)
    return pywt.dwt_max_level(data_len=np.min(shape), filter_len=wv.dec_len)

def scalar_quantizer(data, step_size):
    return np.sign(data)*(abs(data)//step_size)

def scalar_dequantizer(coeffs, step_size, delta=0.5):
    # delta: a user-selectable parameter. 0 <= delta < 1.
    return np.sign(coeffs)*(abs(coeffs) + delta)*step_size

def PSNR(im1, im2):
    # bw imgs
    return 10*np.log10(2**64/(np.sum((im1 - im2)**2)))

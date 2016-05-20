from __future__ import division
from SSTV import *
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
from scipy import misc
from encoding2 import encode_ar, decode_ar, slice_box, compress_np, decompress_np
import itertools as its
from SSTV.wavcompress import getSize
from SSTV.wavcompress import expand_dwt

# Set up
fraction_coeffs = .01
image_fname = 'imgs/lena.bmp'
image = misc.imread(image_fname) # load an image: {dude, cal, lake,...}
original = np.copy(image)
_delta = 0.1 
or_shape = image.shape

# Gaussian Filter [IMPORTANT] -- LPF
image = gf(image, 0.08) 
image = join_imgs([ map2multiple(im, 64) for im in split_img(image)]) # for weird shapes, map2multiple
yccimgs = split_img(rgb2ycc(image))
wavelet='bior4.4'

print('Original shape:',or_shape)
print('New shape:',image.shape)

dwts = []
ds_ycc = []
compressed_imgs = []
us_ycc = []

sample_facts = [0,2,2]

print('Downsampling YCC channels...')
for i in range(len(yccimgs)):
    ds_ycc += [downsample(yccimgs[i],sample_facts[i])]

levels = [max_decomp_level(im.shape, wavelet) for im in ds_ycc]
print('Levels:',levels)

dwt_shapes = np.zeros(len(ds_ycc)*2)

print('Compressing...')
for i in range(len(ds_ycc)):
    numlevels = levels[i]
    stepsize = 2**numlevels
    im = ds_ycc[i]

    # dwt2im, orig_bit_depth = compress_bw(im, fraction_coeffs, step_size=stepsize, wvlt=wavelet, level=numlevels)
    # dwts += [dwt2im]

    dwt2im_big, orig_bit_depth = compress_bw(im, fraction_coeffs, step_size=stepsize, wvlt=wavelet, level=numlevels)
    dwt_shapes[i], dwt_shapes[i+1] = dwt2im_big.shape[0], dwt2im_big.shape[1]
    shrunk_dwt2im = slice_box(dwt2im_big, idx=5)
    dwts += [shrunk_dwt2im]
    dwt2im = expand_dwt(dwts[i], (dwt_shapes[i], dwt_shapes[i+1]))
    
    compressed_img = decompress_bw(dwt2im, im.shape, orig_bit_depth, step_size=stepsize, delta=_delta, wvlt=wavelet, level=numlevels)
    compressed_imgs += [compressed_img]

    if i == 0:
        fraction_coeffs = fraction_coeffs/2

print('Upsampling YCC channels...')
for i in range(len(compressed_imgs)):
    us_ycc += [upsample(compressed_imgs[i],sample_facts[i])]

print('Reconstructing original...')
rec = ycc2rgb(join_imgs(us_ycc))
rec = map2original(rec, or_shape)

# # Without encoding:
print('Saving compressed file...')
full_array = np.array(dwts)

fnames = ['myfile0','myfile1','myfile2']

# Create and compress numpy arrays
compress_np(dwts[0],fnames[0])
compress_np(dwts[1],fnames[1])
compress_np(dwts[2],fnames[2])

# Recreate numpy arrays
dwt_0 = decompress_np(fnames[0])
dwt_1 = decompress_np(fnames[1])
dwt_2 = decompress_np(fnames[2])

loaded = np.array([dwt_0,dwt_1,dwt_2])
print('Are all equal?', np.sum(loaded != full_array)<3)

filesize = 0
for fn in fnames:
    filesize += getSize(fn+'.gz')

print('PSNR:', PSNR(original, rec))
print 'New file size [bytes]:', filesize
print 'Original file size [bytes]:', getSize(image_fname)
print 'Percentage size new/original [%]: ', 100*filesize/getSize(image_fname)

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()






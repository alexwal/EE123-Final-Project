from __future__ import division
from SSTV import *
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
from scipy import misc
from encoding2 import encode_ar, decode_ar
import itertools as its
from SSTV.wavcompress import getSize
from SSTV.wavcompress import expand_dwt

def nonzero_box(m, idx=0):
    # Slices m to a continuous box that extends from the top left corner
    # to each row and column that contains an entry larger than 
    # the idx smallest unqiue entry of m.
    # The function name is a temporary misnomer.

    def threshold(m, idx):
        M = np.sort(np.unique(abs(m.ravel()))) # ordered smallest to biggest
        idx = min(int(idx), len(M)-1)
        thr = M[idx]
        print('Threshhold:',thr)
        return thr

    thr = threshold(m, idx)
    row_zero = [i for i in range(len(m)) if abs(m[i]).max() <= thr and abs(m[i]).min() <= thr]
    col_zero = [i for i in range(len(m.T)) if abs(m.T[i]).max() <= thr and abs(m.T[i]).min() <= thr]
    row_rev = row_zero[::-1]

    r, i = 0, 0
    while i < len(row_rev)-1 and row_rev[i] - row_rev[i+1] == 1:
        i += 1
        r = i
    r = len(row_zero) - r - 1
    col_rev = col_zero[::-1]
    c, i = 0, 0
    while i < len(col_rev)-1 and col_rev[i] - col_rev[i+1] == 1:
        i += 1
        c = i
    c = len(col_zero) - c - 1
    ridx,cidx = None, None
    if r >= 0:
        ridx = row_zero[r]
    if c >= 0:
        cidx = col_zero[c]
    return m[:ridx,:cidx]

# Set up
fraction_coeffs = 1
image_fname = 'imgs/lena.bmp'
image = misc.imread(image_fname) # load an image: {dude, cal, lake,...}
original = np.copy(image)
_delta = 0.1 
or_shape = image.shape

# Gaussian Filter [IMPORTANT] -- LPF
image = gf(image, 0.15) 
image = join_imgs([ map2multiple(im, 64) for im in split_img(image)]) # for weird shapes, map2multiple
yccimgs = split_img(rgb2ycc(image))
wavelet='bior4.4'

print('Original shape:',or_shape)
print('New shape:',image.shape)

dwts = []
ds_ycc = []
compressed_imgs = []
us_ycc = []

sample_facts = [2,4,4]

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
    shrunk_dwt2im = nonzero_box(dwt2im_big, idx=10)
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
f_send = 'WOOF'
kosher.mk_cwd_file(f_send)
full_array = np.array(dwts)
kosher.save(full_array, f_send)
loaded = kosher.load(f_send)
print('Are all equal?', np.sum(loaded != full_array)<3)

print('PSNR:', PSNR(original, rec))
print 'New file size [bytes]:', getSize(f_send)
print 'Original file size [bytes]:', getSize(image_fname)
print 'Percentage size new/original [%]: ', 100*getSize(f_send)/getSize(image_fname)

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()









from __future__ import division
from SSTV import *
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
from scipy import misc

# Set up
fraction_coeffs = 0.04
image = misc.imread('imgs/lena.bmp') # load an image: {dude, cal, lake,...}
original = np.copy(image)
_delta = 0.1
or_shape = image.shape

# Gaussian Filter [IMPORTANT] -- LPF
image = gf(image, 0.2) 
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

print('Compressing...')
for i in range(len(ds_ycc)):
    numlevels = levels[i]
    stepsize = 2**numlevels
    im = ds_ycc[i]
    dwt2im, orig_bit_depth = compress_bw(im, fraction_coeffs, step_size=stepsize, wvlt=wavelet, level=numlevels)
    dwts += [dwt2im]
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

print('PSNR:', PSNR(original, rec))

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()

print('Saving compressed file...')
f_send = 'WOOF'
kosher.mk_cwd_file(f_send)
full_array = np.array(dwts)
kosher.save(full_array, f_send)
loaded = kosher.load(f_send)
print('Are all equal?', np.sum(loaded != full_array)<3)


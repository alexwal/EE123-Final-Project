from __future__ import division
from scipy import misc
from SSTV import *
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
from SSTV.wavcompress import getSize

# Set up
fraction_coeffs = 0.04
image_fname = 'imgs/lena.bmp'  # load an image: {dude, cal, lake,...}
image = misc.imread(image_fname)
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

print('Compressing...')
shapes = []
orig_depths = []
for i in range(len(ds_ycc)):
    numlevels = levels[i]
    stepsize = 2**numlevels
    im = ds_ycc[i]
    dwt2im, orig_bit_depth = compress_bw(im, fraction_coeffs, step_size=stepsize, wvlt=wavelet, level=numlevels)
    dwts += [dwt2im]
    shapes.append(im.shape)
    orig_depths.append(orig_bit_depth)
    if i == 0:
        fraction_coeffs = fraction_coeffs/2

print('Saving compressed file...')
f_send = 'compressed'
kosher.mk_cwd_file(f_send)
full_array = np.array(dwts)
kosher.save(full_array, f_send)

blocks = []
with open(f_send, 'rb') as fd:
    while True:
        bytes_read = fd.read(16)
        if not bytes_read:
            break
        b = bitarray.bitarray()
        b.frombytes(bytes_read)
        blocks.append(bitarray.bitarray(encode.bit_stuff(b)))

print 'Constructing packets...'
packets = []
prefix = [0,1,1,1,1,1,1,0]*10
for block in blocks:
    packets.append(bitarray.bitarray(prefix + block.tolist() + prefix))

print(len(packets)), " packets."

## AFSK modulate packets
print 'Message construction...'
# TODO: Note that this buffer may need a better design...
buffer_bits = [0,1]*10
full_msg = bitarray.bitarray(buffer_bits)
for packet in packets:
   full_msg.extend(NRZ2NRZI(packet))
   full_msg.extend(buffer_bits)

print 'AFSK modulation...'
full_signal = afsk.afsk1200(full_msg)

## AFSK demodulate signal
print 'AFSK demodulation...'
demod_sig = afsk.nc_afsk1200Demod(full_signal)
print '    Running PLL...'
sample_idx = afsk.PLL(demod_sig)
samples = demod_sig[sample_idx]
bitstream = bitarray.bitarray((samples>0).tolist())
NRZ = NRZI2NRZ(bitstream)
packets_rec = afsk.findPackets(NRZ)

# Recover bitstreams from packets, bit unstuff
print 'Decoding packets...'
for i in range(len(packets_rec)):
    packets_rec[i] = afsk.deflag_packet(packets_rec[i])
    packets_rec[i] = encode.bit_unstuff(packets_rec[i])

print(len(packets_rec)), " packets recovered."

# packets to bytes
inbytes = [pr.tobytes() for pr in packets_rec]
f_recv = 'received'
kosher.mk_cwd_file(f_recv)
with open(f_recv, 'wb') as fd:
    fd.write(''.join(inbytes))

# Reconstruction
# TODO: shapes, orig_depths, levels need to be transmitted eventually?
loaded = kosher.load(f_recv)
for i in range(len(loaded)):
    dwt2im = loaded[i]
    shape = shapes[i]
    orig_bit_depth = orig_depths[i]
    numlevels = levels[i]
    stepsize = 2**numlevels
    compressed_img = decompress_bw(dwt2im, shape, orig_bit_depth, step_size=stepsize, delta=_delta, wvlt=wavelet, level=numlevels)
    compressed_imgs += [compressed_img]

print('Upsampling YCC channels...')
for i in range(len(compressed_imgs)):
    us_ycc += [upsample(compressed_imgs[i],sample_facts[i])]

print('Reconstructing original...')
rec = ycc2rgb(join_imgs(us_ycc))
rec = map2original(rec, or_shape)

print('PSNR:', PSNR(original, rec))
print 'New file size [bytes]:', getSize(f_send)
print 'Original file size [bytes]:', getSize(image_fname)
print 'Percentage size new/original [%]: ', 100*getSize(f_send)/getSize(image_fname)

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()






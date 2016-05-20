from __future__ import division
from scipy import misc
from SSTV import *
from SSTV.wavcompress import getSize
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
import aprs
from aprs import *

import matplotlib.cbook as cbook
from matplotlib import lines as mpl_lines

plt.rcParams['figure.figsize'] = (16, 8)

# start the recording and playing threads
dusb_in = 2
dusb_out = 2
callsign = "KM6BLD"
fs = 11025

modem = aprs.TNCaprs(fs = fs ,Abuffer = 1024,Nchunks = 12)

Qin = Queue.Queue()
cQin = Queue.Queue()
p = pyaudio.PyAudio()

t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin,p, fs, dusb_in))
t_rec.start()

starttime = time.time()
npack = 0
state = 0

while(1):
    tmp = Qin.get()
    packets  = modem.processBuffer(tmp)
    for ax in packets:
        npack = npack + 1
        print((str(npack)+")",str(ax)))
        if state == 0 and ax.destination[:5]=="BEGIN":
            f1 = open("recv", "wb")
            state = 1
        elif state == 1 and ax.destination[:3] == "END": 
            state = 2
            break
        elif state == 1:
            f1.write(ax.info)
    if state == 2 :
        break

print('Finished recording.')
transmission_time = time.time() - starttime
print 'Recording time: ', transmission_time
cQin.put("EOT")
f1.close()

# Reconstruction
f_recv = "recv"
loaded = kosher.load(f_recv)
rec_shapes = loaded[0]
rec_orig_depths = loaded[1]
rec_levels = loaded[2]
rec_sample_facts = loaded[3]
or_shape = loaded[4]
wavelet = loaded[5]

_delta = 0.1 #for now
compressed_imgs = []
us_ycc = []

print('Decompressing...')
for i in range(6,len(loaded)):
    print('      ... '+str(i-6))
    dwt2im = loaded[i]
    shape = rec_shapes[i-6]
    orig_bit_depth = rec_orig_depths[i-6]
    numlevels = rec_levels[i-6]
    stepsize = 2**numlevels
    compressed_img = decompress_bw(dwt2im, shape, orig_bit_depth, step_size=stepsize, delta=_delta, wvlt=wavelet, level=numlevels)
    compressed_imgs += [compressed_img]

print('Upsampling YCC channels...')
for i in range(len(compressed_imgs)):
    us_ycc += [upsample(compressed_imgs[i], rec_sample_facts[i])]

print('Reconstructing original...')
rec = ycc2rgb(join_imgs(us_ycc))
rec = map2original(rec, or_shape)

image_file = 'imgs/lena.tiff'
image = misc.imread(image_file) # load an image: {dude, cal, lake,...}
original = np.copy(image)

print 'Recording time: ', transmission_time
print 'PSNR:', PSNR(original, rec)

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()


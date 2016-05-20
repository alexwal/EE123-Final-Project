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

class ComparePlot:
    def __init__(self, original, compressed, fig=None, ax=None, img=None):
        self.fig = plt.get_current_fig_manager().canvas.figure
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.original = original
        self.compressed = compressed
        plt.rcParams['figure.figsize'] = (16, 8)
        plt.title('Comparison: left original, right compressed')
        plt.imshow(np.concatenate((self.original, self.compressed), axis=1))

    def show(self):
        plt.show()

    def onKey(self, event):
        if event.key == 'y':
            print('continue')
            plt.close()
        if event.key == 'n':
            print('exiting')
            plt.close()
            exit(0)
        event.canvas.draw()


# Set up
fraction_coeffs = 0.08
sample_facts = [2,4,4]
image_file = 'imgs/lena.tiff'
image = misc.imread(image_file) # load an image: {dude, cal, lake,...}
original = np.copy(image)
_delta = 0.5
or_shape = image.shape

# Gaussian Filter [IMPORTANT] -- LPF
image = gf(image, 0.1)
image = join_imgs([ map2multiple(im, 64) for im in split_img(image)]) # for weird shapes, map2multiple
yccimgs = split_img(rgb2ycc(image))
wavelet='bior4.4'
#wavelet='haar'

print('Original shape:',or_shape)
print('New shape:',image.shape)

dwts = []
ds_ycc = []
compressed_imgs = []
us_ycc = []


print('Downsampling YCC channels...')
for i in range(len(yccimgs)):
    ds_ycc += [downsample(yccimgs[i],sample_facts[i])]

levels = [min(3, max_decomp_level(im.shape, wavelet)) for im in ds_ycc]
print('Levels:',levels)

print('Compressing...')
shapes = []
orig_depths = []
compressed_imgs = []
for i in range(len(ds_ycc)):
    numlevels = levels[i]
    stepsize = 2**numlevels
    im = ds_ycc[i]
    dwt2im, orig_bit_depth = compress_bw(im, fraction_coeffs, step_size=stepsize, wvlt=wavelet, level=numlevels)
    dwts += [dwt2im]
    
    # to check if img okay
    compressed_img = decompress_bw(dwt2im, im.shape, orig_bit_depth, step_size=stepsize, delta=_delta, wvlt=wavelet, level=numlevels)
    compressed_imgs += [compressed_img]

    shapes.append(im.shape)
    orig_depths.append(orig_bit_depth)
#    if i == 0:
#        fraction_coeffs = fraction_coeffs/2

# Check if image is okay: [START]
ds_ycc = []
us_ycc = []
for i in range(len(yccimgs)):
    ds_ycc += [downsample(yccimgs[i],sample_facts[i])]
for i in range(len(compressed_imgs)):
    us_ycc += [upsample(compressed_imgs[i],sample_facts[i])]
rectest = ycc2rgb(join_imgs(us_ycc))
rectest = map2original(rectest, or_shape)
print('TEST PSNR:', PSNR(original, rectest))
fig = plt.figure()
ax = plt.subplot(111)
cp = ComparePlot(original, rectest, fig=fig, ax=ax)
cp.show()
# Check if image is okay: [END]

us_ycc = []
ds_ycc = []
compressed_imgs = []

# Save metadata:
dwts.insert(0, wavelet)
dwts.insert(0, or_shape)
dwts.insert(0, sample_facts) # Sample facts fourth
dwts.insert(0, levels) # Levels is in the third position
dwts.insert(0, orig_depths) # Orig depths is in the second position
dwts.insert(0, shapes) # Shapes in the front

print('Saving compressed file...')
f_send = 'to_send'
kosher.mk_cwd_file(f_send)
full_array = np.array(dwts)
kosher.save(full_array, f_send)

##

dusb_out = 2
callsign = "KM6BLD"
fname = "to_send"
f = open(fname,"rb")
fs = 11025

if sys.platform == 'darwin':  # Mac
    s = serial.Serial(port='/dev/tty.SLAB_USBtoUART')
else:                         #windows
    s = serial.Serial(port='COM1') # CHANGE !!!!!!

modem = aprs.TNCaprs(fs = fs ,Abuffer = 1024,Nchunks = 12)

Qout = Queue.Queue()
cQout = Queue.Queue()

# create a pyaudio object
p = pyaudio.PyAudio()

print("Putting packets in Queue")
gain = 1.0
npp = 1
tmp = modem.modulatPacket(callsign, "", "BEGIN", fname , preflags=10, postflags=2)
Qout.put(gain*tmp)
while(1):
    bytes = f.read(256)    
    tmp = modem.modulatPacket(callsign, "", str(npp), bytes, preflags=4, postflags=2)    
    Qout.put(gain*tmp)
    npp = npp+1
    if len(bytes) < 256:
        break
tmp = modem.modulatPacket(callsign, "", "END", "This is the end of transmission", preflags=2, postflags=80)

Qout.put(gain*tmp)
Qout.put("EOT")
f.close()

print("Done generating packets")
starttime = time.time()

play_audio(Qout, cQout, p, fs, dusb_out, keydelay = 0.3)

print 'Transmission time', time.time() - starttime



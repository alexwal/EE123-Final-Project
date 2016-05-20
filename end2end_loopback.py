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
fraction_coeffs = 1.0
image_file = 'imgs/cal.tiff'
image = misc.imread(image_file) # load an image: {dude, cal, lake,...}
original = np.copy(image)
_delta = 0.1
or_shape = image.shape

# Gaussian Filter [IMPORTANT] -- LPF
image = gf(image, 0.2)
image = join_imgs([ map2multiple(im, 64) for im in split_img(image)]) # for weird shapes, map2multiple
yccimgs = split_img(rgb2ycc(image))
#wavelet='bior4.4'
wavelet='haar'

print('Original shape:',or_shape)
print('New shape:',image.shape)

dwts = []
ds_ycc = []
compressed_imgs = []
us_ycc = []

sample_facts = [0,0,0]

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
dwts.insert(0, or_shape)
dwts.insert(0, sample_facts) # Sample facts fourth
dwts.insert(0, levels) # Levels is in the third position
dwts.insert(0, orig_depths) # Orig depths is in the second position
dwts.insert(0, shapes) # Shapes in the front

print('Saving compressed file...')
f_send = 'WOOF'
kosher.mk_cwd_file(f_send)
full_array = np.array(dwts)
kosher.save(full_array, f_send)

##

dusb_in = 2
dusb_out = 2
callsign = "FUCK6"
fname = "WOOF"
f = open(fname,"rb")
fs = 11025

modem = aprs.TNCaprs(fs = fs ,Abuffer = 1024,Nchunks = 12)

Qin = Queue.Queue()
Qout = Queue.Queue()

# create a control fifo to kill threads when done
cQin = Queue.Queue()
cQout = Queue.Queue()

# create a pyaudio object
p = pyaudio.PyAudio()

# initialize a recording thread. 
t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin,p, fs, dusb_in))
t_play = threading.Thread(target = play_audio,   args = (Qout, cQout,p, fs, dusb_out))


print("Putting packets in Queue")
npp = 0
sig = 0
tmp = modem.modulatPacket(callsign, "", "BEGIN", fname , preflags=80, postflags=2, sigma=sig) 
Qout.put(tmp)
while(1):
    bytes = f.read(256)    
    tmp = modem.modulatPacket(callsign, "", str(npp), bytes, preflags=4, postflags=2, sigma=sig)    
    Qout.put(tmp)
    npp = npp+1
    if len(bytes) < 256:
        break
tmp = modem.modulatPacket(callsign, "", "END", "This is the end of transmission", preflags=2, postflags=80, sigma=sig)
Qout.put(tmp)
Qout.put("EOT")
f.close()

print("Done generating packets")

# start the recording and playing threads
t_rec.start()
time.sleep(2)
t_play.start()

starttime = time.time()
npack = 0
state = 0

while(1):
    tmp = Qin.get()
    Qout.put(tmp)
    packets  = modem.processBuffer(tmp)
    for ax in packets:
        npack = npack + 1
        print((str(npack)+")",str(ax)))
        if state == 0 and ax.destination[:5]=="BEGIN":
            f1 = open("rec_"+ax.info,"wb")
            state = 1
        elif state == 1 and ax.destination[:3] == "END": 
            state = 2  
            break
        elif state == 1:
            f1.write(ax.info)
    if state == 2 :
        break

transmission_time = time.time() - starttime
cQout.put("EOT")
cQin.put("EOT")
f1.close()

##

# Reconstruction
# TODO: shapes, orig_depths, levels need to be transmitted eventually?
f_recv = 'rec_WOOF'
loaded = kosher.load(f_recv)
rec_shapes = loaded[0]
rec_orig_depths = loaded[1]
rec_levels = loaded[2]
rec_sample_facts = loaded[3]


for i in range(5,len(loaded)):
    dwt2im = loaded[i]
    shape = rec_shapes[i-5]
    orig_bit_depth = rec_orig_depths[i-5]
    numlevels = rec_levels[i-5]
    stepsize = 2**numlevels
    compressed_img = decompress_bw(dwt2im, shape, orig_bit_depth, step_size=stepsize, delta=_delta, wvlt=wavelet, level=numlevels)
    compressed_imgs += [compressed_img]

print('Upsampling YCC channels...')
for i in range(len(compressed_imgs)):
    us_ycc += [upsample(compressed_imgs[i], rec_sample_facts[i])]

print('Reconstructing original...')
rec = ycc2rgb(join_imgs(us_ycc))
rec = map2original(rec, loaded[4])

print 'Transmission time: ', transmission_time
print 'PSNR:', PSNR(original, rec)
print 'New file size [bytes]:', getSize(f_send)
print 'Original file size [bytes]:', getSize(image_file)
print 'Percentage size new/original [%]: ', 100*getSize(f_send)/getSize(image_file)

plt.rcParams['figure.figsize'] = (16, 8)
compare_imgs(original, rec)
plt.show()


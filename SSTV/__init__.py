# Alex Walczak, Diego Kiner Spring 2016
# EE 123 SSTV project
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pyaudio, Queue, threading,time, sys, threading,time, serial
from numpy import pi, sin, zeros, r_
from scipy import signal
from rtlsdr import RtlSdr
from scipy import misc
import cPickle as pickle
import operator
import bitarray, time, urllib, ssl
from scipy import signal, integrate
from fractions import gcd
import pywt
import bitarray, time, urllib, ssl
# import ax25

from wavcompress import compress_bw, decompress_bw, PSNR, max_decomp_level, map2multiple, map2original
from encode import array2code_blocks, code_blocks2array, bitarray2block, block2bitarray
from color import rgb2ycc, ycc2rgb, rgb2gray, join_imgs, split_img, compare_imgs, clean8bit

np.random.seed(666)
print('\nAlex and Diego\'s project code successfully loaded!\n')

# # # Serialize object
# f = open('mypickle.pickle', 'wb')
# pickle.dump(myobject, f, protocol=pickle.HIGHEST_PROTOCOL)
# f.close()

# # # Load object
# f = open('mypickle.pickle', 'rb')
# mypickledobject = pickle.load(f)
# f.close()
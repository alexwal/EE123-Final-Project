from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Converting color spaces

def rgb2ycc(img):
    # also convert to 8-bit
    rgb = np.copy(img)
    mtx = np.matrix(
        [[  0.299  ,   0.587   ,   0.114  ],
         [ -0.169  ,  -0.331   ,   0.499  ],
         [  0.499  ,  -0.418   ,  -0.0813 ]])
    return np.array(rgb.reshape(-1,3) * mtx.T + np.array([0, 128, 128])).reshape(rgb.shape)

def ycc2rgb(img):
    # inverse of above
    ycc = np.copy(img)
    mtx = np.matrix(
         [[ 1   ,    0      ,    1.402  ],
          [ 1   ,   -0.344  ,   -0.714  ],
          [ 1   ,   +1.772  ,    0      ]])
    ycc[:,:,1] -= 128
    ycc[:,:,2] -= 128
    return np.uint8(np.array(ycc.reshape(-1,3) * mtx.T).reshape(ycc.shape))

def split_img(img):
    # into 3 separate channels
    return img[:,:,0], img[:,:,1], img[:,:,2]

def join_imgs(imgs):
    # inverse of above
    imshp = imgs[0].shape
    joined = np.zeros((imshp[0], imshp[1], len(imgs)))
    for i in range(len(imgs)):
        joined[:,:,i] = imgs[i]
    return joined

def compare_imgs(original, compressed):
    plt.figure()
    plt.title('Comparison: left original, right compressed')
    plt.imshow(np.concatenate((original, compressed), axis=1), cmap = 'gray')

def rgb2gray(img):
    # converts RGB img into 1 channel
    # mainly for testing
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

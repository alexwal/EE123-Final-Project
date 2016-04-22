from __future__ import division
import numpy as np
from wavcompress import image2blocks, blocks2image

# Transmission and encoding code

def array2code_blocks(array, code_block_size=16):
    # We will do processing on code blocks, and encode code blocks and send them specially (entropy enocding)
    # code_block_size is a power of 2
    # code blocks are sections of the wavelet domain
    return image2blocks(array, code_block_size)

def code_blocks2array(code_blocks, original_shape, block_size):
    # possible issue: int8 call
    return np.int8(blocks2image(code_blocks, int(original_shape[1]/block_size), block_size))

def int2bitarray(intarray, width=8):
    # converts an array of integers type uint8 and returns an equivalent Python bitarray.bitarray
    # use width = 1 if intarray contains 1, 0s
    # use width = 8 for 8 bit ints
    bits = ''
    for num in intarray:
        bits += np.binary_repr(num,width)
    return bitarray.bitarray(bits)

def encode_block(block, width):
    #todo: actual encoding.
    array_of_bits = []
    for row in block:
        array_of_bits += [int2bitarray(row,width)]
    return array_of_bits

def decode_block(block):
    #todo
    return

def zigzag(n):
    # returns the order of zigzag traversal of a square nxn matrix
    # borrowed code from http://paddy3118.bnp.logspot.com/2008/08/zig-zag.html
    # and from http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    def sortbyvalue(x):
        return np.array(sorted(x.items(), key=operator.itemgetter(1)))[:,0]
    indexorder = sorted(((x,y) for x in range(n) for y in range(n)),\
                        key = lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]) ) 
    return sortbyvalue(dict((index,n) for n,index in enumerate(indexorder)))

def block_to_zigzag(block):
    # flattens a square block to its zigzag traversal
    inds = zigzag(block.shape[0])
    zz = []
    for ind in inds:
        zz += [block[ind]]
    return zz

def zigzag_to_block(zz):
    # the inverse of the above
    inds = zigzag(int(np.sqrt(len(zz))))
    block = np.zeros((int(np.sqrt(len(zz))),int(np.sqrt(len(zz)))))
    for i in range(len(zz)):
        block[inds[i]] = zz[i]
    return block

def num_nonoverlap_blocks(shape, block_size):
    # shape: image shape, (rows, cols), rows and cols both must be at least block_size**2.
    # block_size: number of pixels in an edge of a block. must be power of 2.
    # returns: num blocks that fit image with shape shape without overlap
    assert np.log(block_size)/np.log(2) % 1 == 0, 'Block size must be a power of 2.'
    rows, cols = shape
    assert rows >= block_size and cols >= block_size, 'Block size too large.'
    horiz = rows//block_size
    vert = cols//block_size
    return horiz, vert

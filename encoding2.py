from __future__ import division
import itertools as its
import numpy as np
def encode_ar(ar):
    # 1d array
    # slice into 256 len slices
    def format_slice(s):
        if len(np.nonzero(s)[0]) == 0:
            return np.array([0,0])
        else:
            return np.append([len(np.nonzero(s)[0])]*1,np.concatenate( [np.nonzero(s)[0], s[s != 0]] ))
    z = 256
    num_zeros = lambda length: (length//z + 1)*z - length # helps pad array to next multiple of 256
    ar = np.hstack([ar, np.zeros(num_zeros(len(ar)))])
    slices = np.array_split(ar, len(ar)/z)
    encoded = [format_slice(s) for s in slices ]
    return np.array(list(its.chain(*encoded)))

def decode_ar(ar):
    # 1d array
    # reslice into 256 len slices
    z = 256
    def rebuild_slice(coords, vals):
        s = np.zeros(z)
        s[np.uint8(coords)] = vals
        return s
    result = []
    i = 0
    while i < len(ar):
        length = int(ar[i])
        i+=1
        if ar[i] == 0 and length == 0:
            print('woof')
            result += [np.zeros(z)]
            i += 1
            continue
        inds = ar[i:i+length]
        i+=length
        vals = ar[i:i+length]
        i += length
        result += [rebuild_slice(inds, vals)]
    return np.array(list(its.chain(*result)))#[:length-z]

def compress_np(array,fname):
    # takes an np array and converts it into a text file
    # gzips the text file
    # text file saved as 'fname'
    # compressed text file saved as 'fname.gz'
    np.savetxt(fname, array, fmt='%i')
    import shutil, gzip
    with open(fname, 'rb') as f_in, gzip.open(fname+'.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def decompress_np(fname):
    # inverse of compress_txt
    # decompressed text file saved as 'fname_'
    import shutil, gzip
    with gzip.open(fname+'.gz', 'rb') as f_in, open(fname+'_', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return np.loadtxt(fname+'_', dtype=np.int8)

def slice_box(m, idx=0):
    # Slices m to a continuous box that extends from the top left corner
    # to each row and column that contains an entry larger than 
    # the idx smallest unqiue entry of m.
    # The function name is a temporary misnomer [fixed].

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
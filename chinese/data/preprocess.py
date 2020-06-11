# Adapted from https://github.com/integeruser/CASIA-HWDB1.1-cnn

import os
import struct
import sys
import h5py
from collections import defaultdict
import numpy as np
from PIL import Image

CASIA_TRN_SIZE = 897758
CASIA_TST_SIZE = 223991
HIT_TRN_SIZE = 367990  # 3755*98
HIT_TST_SIZE = 90120  # 3755*24
NUM_CLASSES = 3755
HEIGHT = 64
WIDTH = 64
LOG_INTERVAL = 100000

def read_gnt_in_directory(gnt_dirpath):
    def samples(f):
        header_size = 10

        # read samples from f until no bytes remaining
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break

            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            assert header_size + width*height == sample_size

            bitmap = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield bitmap, tagcode

    for file_name in sorted(os.listdir(gnt_dirpath)):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples(f):
                    if tagcode >= 0xB0A1 and tagcode <= 0xF7FE:  # only yield Chinese characters
                        yield bitmap, tagcode
                    
def normalize_bitmap(bitmap):
    # pad the bitmap to make it squared
    pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
    if bitmap.shape[0] < bitmap.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    pad = 4
    bitmap = np.array(Image.fromarray(bitmap).resize((HEIGHT - pad*2, WIDTH - pad*2)))
    bitmap = np.lib.pad(bitmap, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
    assert bitmap.shape == (HEIGHT, WIDTH)

    bitmap = np.expand_dims(bitmap, axis=0)
    assert bitmap.shape == (1, HEIGHT, WIDTH)
    return bitmap

def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')

def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]

def read_labels():
    labels_path='character/labels'
    with open(labels_path, 'rb') as f:
        header_size = 3
        header = np.fromfile(f, dtype='uint8', count=header_size)
        sample_size = header[0] + (header[1]<<8)
        char_len = header[2]
        assert char_len == 2

        tagcodes = []
        # read samples from f until no bytes remaining
        while True:
            character = np.fromfile(f, dtype='uint8', count=char_len)
            if not character.size: break
            tagcode = character[1] + (character[0]<<8)
            tagcodes.append(tagcode)
        assert len(tagcodes) == sample_size
        
    return tagcodes

# generate HWDB1.1.hdf5
trn_dirpath = 'Gnt1.1Train/'
tst_dirpath = 'Gnt1.1Test/'

tagcode_to_label = dict()  # tagcode -> int
tagcodes = read_labels()
for i in range(62, 3817):  # GB2312-80 level-1 encoding, or GB1
    t = tagcodes[i]
    i = i - 62
    tagcode_to_label[t] = i
assert len(tagcode_to_label) == NUM_CLASSES

TRN_SPLIT = 200  # for each character, the first 200 images are in trn set

with h5py.File('HWDB1.1.hdf5', 'w') as f:
    print('Processing \'Train\'...')
    
    trn_size = NUM_CLASSES * TRN_SPLIT  # for each character, first TRN_SPLIT images are in trn set
    trn_grp = f.create_group('trn')
    trn_x = trn_grp.create_dataset('x', (trn_size, 1, HEIGHT, WIDTH), dtype='uint8')
    trn_y = trn_grp.create_dataset('y', (trn_size, 1), dtype='uint16')
    
    vld_size = CASIA_TRN_SIZE - trn_size
    vld_grp = f.create_group('vld')
    vld_x = vld_grp.create_dataset('x', (vld_size, 1, HEIGHT, WIDTH), dtype='uint8')
    vld_y = vld_grp.create_dataset('y', (vld_size, 1), dtype='uint16')
    
    print('Total train:', CASIA_TRN_SIZE)
    print('trn x shape:', (trn_size, 1, HEIGHT, WIDTH))
    print('trn y shape:', (trn_size, 1))
    print('val x shape:', (vld_size, 1, HEIGHT, WIDTH))
    print('val y shape:', (vld_size, 1))
    
    tagcode_to_count = defaultdict(int)
    trn_i = vld_i = 0
    for i, (bitmap, tagcode) in enumerate(read_gnt_in_directory(trn_dirpath)):
        if i % LOG_INTERVAL == 0:
            print(i)
        is_trn = tagcode_to_count[tagcode] < TRN_SPLIT
        tagcode_to_count[tagcode] += 1
        bitmap  = normalize_bitmap(bitmap)
        label = tagcode_to_label[tagcode]
        if is_trn:
            trn_x[trn_i] = bitmap
            trn_y[trn_i] = label
            trn_i += 1
        else:
            vld_x[vld_i] = bitmap
            vld_y[vld_i] = label
            vld_i += 1
    print(i+1)
    assert i+1 == CASIA_TRN_SIZE
    assert trn_i == trn_size
    assert vld_i == vld_size

    print('Processing \'Test\'...')

    tst_grp = f.create_group('tst')
    tst_x = tst_grp.create_dataset('x', (CASIA_TST_SIZE, 1, HEIGHT, WIDTH), dtype='uint8')
    tst_y = tst_grp.create_dataset('y', (CASIA_TST_SIZE, 1), dtype='uint16')
    
    print('tst x shape:', (CASIA_TST_SIZE, 1, HEIGHT, WIDTH))
    print('tst y shape:', (CASIA_TST_SIZE, 1))

    for i, (bitmap, tagcode) in enumerate(read_gnt_in_directory(tst_dirpath)):
        if i % LOG_INTERVAL == 0:
            print(i)
        bitmap  = normalize_bitmap(bitmap)
        label = tagcode_to_label[tagcode]
        tst_x[i] = bitmap
        tst_y[i] = label
    print(i+1)
    assert i+1 == CASIA_TST_SIZE
    
# generate HIT_OR3C.hdf5
def read_images_in_directory(images_dirpath, tagcodes):
    def samples(f):
        header_size = 6

        header = np.fromfile(f, dtype='uint8', count=header_size)
        sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
        width = header[4] + 0
        height = header[5] + 0

        # read samples from f until no bytes remaining
        i = -1
        while True:
            i += 1
            bitmap = np.fromfile(f, dtype='uint8', count=width*height)
            if not bitmap.size:
                assert i == sample_size
                break
            if i < 62 or i > 3816:
                continue
            bitmap = bitmap.reshape((height, width))
            yield bitmap, tagcodes[i]

    for file_name in sorted(os.listdir(images_dirpath)):
        if file_name.endswith('images'):
            file_path = os.path.join(images_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples(f):
                    yield bitmap, tagcode

trn_dirpath = 'character/trn/'
tst_dirpath = 'character/tst/'

TRN_SPLIT = 80

with h5py.File('HIT_OR3C.hdf5', 'w') as f:
    print('Processing \'Train\'...')
    
    trn_size = NUM_CLASSES * TRN_SPLIT  # for each character, first TRN_SPLIT images are in trn set
    trn_grp = f.create_group('trn')
    trn_x = trn_grp.create_dataset('x', (trn_size, 1, HEIGHT, WIDTH), dtype='uint8')
    trn_y = trn_grp.create_dataset('y', (trn_size, 1), dtype='uint16')
    
    vld_size = HIT_TRN_SIZE - trn_size
    vld_grp = f.create_group('vld')
    vld_x = vld_grp.create_dataset('x', (vld_size, 1, HEIGHT, WIDTH), dtype='uint8')
    vld_y = vld_grp.create_dataset('y', (vld_size, 1), dtype='uint16')
    
    print('Total train:', HIT_TRN_SIZE)
    print('trn x shape:', (trn_size, 1, HEIGHT, WIDTH))
    print('trn y shape:', (trn_size, 1))
    print('val x shape:', (vld_size, 1, HEIGHT, WIDTH))
    print('val y shape:', (vld_size, 1))
    
    tagcode_to_count = defaultdict(int)
    trn_i = vld_i = 0
    for i, (bitmap, tagcode) in enumerate(read_images_in_directory(trn_dirpath, tagcodes)):
        if i % LOG_INTERVAL == 0:
            print(i)
        is_trn = tagcode_to_count[tagcode] < TRN_SPLIT
        tagcode_to_count[tagcode] += 1
        bitmap  = normalize_bitmap(bitmap)
        label = tagcode_to_label[tagcode]
        if is_trn:
            trn_x[trn_i] = bitmap
            trn_y[trn_i] = label
            trn_i += 1
        else:
            vld_x[vld_i] = bitmap
            vld_y[vld_i] = label
            vld_i += 1
    print(i+1)
    assert i+1 == HIT_TRN_SIZE
    assert trn_i == trn_size
    assert vld_i == vld_size

    print('Processing \'Test\'...')

    tst_grp = f.create_group('tst')
    tst_x = tst_grp.create_dataset('x', (HIT_TST_SIZE, 1, HEIGHT, WIDTH), dtype='uint8')
    tst_y = tst_grp.create_dataset('y', (HIT_TST_SIZE, 1), dtype='uint16')
    
    print('tst x shape:', (HIT_TST_SIZE, 1, HEIGHT, WIDTH))
    print('tst y shape:', (HIT_TST_SIZE, 1))

    for i, (bitmap, tagcode) in enumerate(read_images_in_directory(tst_dirpath, tagcodes)):
        if i % LOG_INTERVAL == 0:
            print(i)
        bitmap  = normalize_bitmap(bitmap)
        label = tagcode_to_label[tagcode]
        tst_x[i] = bitmap
        tst_y[i] = label
    print(i+1)
    assert i+1 == HIT_TST_SIZE

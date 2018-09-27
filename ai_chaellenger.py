from dsl_data import aichanellger
from dsl_data.bdd import BDD_AREA
import random
import numpy as np
def get_leaf(batch_size,is_shuff = True,image_size=512):
    data_set = aichanellger.AiCh('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/fenge',image_size=512)
    idx = list(range(data_set.len()))
    print(data_set.len())
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, mask = data_set.pull_item(idx[index])
            except:
                index = index+1
                print(index)
                continue
            img = img - [123.15, 115.90, 103.06]
            mask = mask/255
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size,image_size,3],dtype=np.float32)
                masks = np.zeros(shape=[batch_size, image_size, image_size], dtype=np.int)
                images[b,:,:,:] = img
                masks[b,:,:] = mask
                b=b+1
                index = index + 1
            else:
                images[b, :, :, :] = img
                masks[b, :, :] = mask
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,masks]
                b = 0

            if index>= data_set.len():
                index = 0

def get_drive(batch_size,is_shuff = True,image_size=None):
    if image_size is None:
        image_size = [768, 1280]
    image_dr = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/train'
    js_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/label/labels/bdd100k_labels_images_train.json'
    data_set = BDD_AREA(js_file=js_file, image_dr=image_dr, image_size = image_size)
    idx = list(range(data_set.len()))
    print(data_set.len())
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
            try:
                img, mask = data_set.pull_item(idx[index])
            except:
                index = index+1
                print(index)
                continue
            img = (img - [123.15, 115.90, 103.06])/1.0
            mask = mask/255
            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                masks = np.zeros(shape=[batch_size, image_size[0], image_size[1],2], dtype=np.int)
                images[b,:,:,:] = img
                masks[b,:,:] = mask
                b=b+1
                index = index + 1
            else:
                images[b, :, :, :] = img
                masks[b, :, :] = mask
                b = b + 1
                index = index + 1
            if b>=batch_size:
                yield [images,masks]
                b = 0

            if index>= data_set.len():
                index = 0

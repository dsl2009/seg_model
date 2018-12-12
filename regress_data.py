import glob
import json
import os
from skimage import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa
import imgaug as ia
labels = ['land', 'canal', 'pond', 'tree', 'other', 'building']
class BigLand(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.data = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.json')
        self.class_mapix = dict(zip(labels,range(len(labels))))
        self.aug = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                rotate=(-30, 30),
            ),
        ])
    def len(self):
        return len(self.data)
    def pull_item(self,idx):
        json_pth = self.data[idx]
        image_pth = json_pth.replace('.json','.png')
        ig_data = io.imread(image_pth)
        instance_masks = np.zeros(shape=[self.image_size[0], self.image_size[1], 6], dtype=np.uint8)
        js_data = json.loads(open(json_pth).read())
        ag = self.aug.to_deterministic()
        ig_data = ag.augment_image(ig_data)
        for b in js_data['boundary']:
            label = b['correctionType']
            points = b['points']
            p = []
            for pp in points:
                p.append([pp['pix_x'],pp['pix_y']])
            label_id = self.class_mapix[label]
            direct = np.zeros(shape=[self.image_size[0], self.image_size[1], 3], dtype=np.uint8)
            cv2.fillPoly(direct, np.asarray([p], np.int), (255, 255, 255))
            instance_masks[:,:,label_id] += direct[:,:,0]

        instance_masks = ag.augment_image(instance_masks)

        return ig_data, instance_masks



def tt():
    d = BigLand([256,256])
    for x in range(100):
        img, mask = d.pull_item(x)
        plt.imshow(img)
        plt.show()
        plt.imshow(mask[:,:,1])
        plt.show()

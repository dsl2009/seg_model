import tensorflow as tf
from loss import get_loss,sigmoid_cross_entropy_balanced,nn_loss
from matplotlib import pyplot as plt
from tensorflow.contrib import slim
from ai_chaellenger import get_drive
import config
import numpy as np
from dsl_data import utils
from skimage import io
import glob
import time
from model import resnet50
import config
def run():
    batch_size = 1
    image = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 3))

    global_step = tf.train.get_or_create_global_step()
    out_put = resnet50.fpn(image)
    out_put = tf.nn.sigmoid(out_put)
    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'drive/model.ckpt-53581')
        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/BDD100K/bdd100k/images/100k/val/*.*'):
            ig = io.imread(x)
            org, window, scale, padding, crop = utils.resize_image_fixed_size(ig, config.image_size)
            ig = org - [123.15, 115.90, 103.06]

            ig = np.expand_dims(ig, 0)

            fd = {image: ig}
            o_put = sess.run(out_put, feed_dict=fd)

            plt.subplot(221)
            plt.title('step1')
            plt.imshow(o_put[0, :, :, 0], aspect="auto", cmap='gray')
            plt.subplot(222)
            plt.title('step2')
            plt.imshow(o_put[0, :, :, 1], aspect="auto", cmap='gray')
            plt.subplot(223)
            plt.title('org')
            plt.imshow(org, aspect="auto")
            plt.show()



if __name__ == '__main__':
    run()
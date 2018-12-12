import tensorflow as tf
from dsl_seg import create_model
from matplotlib import pyplot as plt
from tensorflow.contrib import slim
from ai_chaellenger import get_land
import config
import numpy as np
import time
from dsl_data import data_loader_multi
def run():
    batch_size = 4
    image = tf.placeholder(dtype=tf.float32, shape=(batch_size, config.image_size[0], config.image_size[1], 3))
    mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, config.image_size[0], config.image_size[1], 6))
    global_step = tf.train.get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=100000,
        decay_rate=0.7,
        staircase=True)

    train_tensors,out_put = create_model(image, mask)
    out_put = tf.nn.sigmoid(out_put)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    '''
       vbs = []
    for s in slim.get_variables():
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name and 'GroupNorm' not in s.name:
            vbs.append(s)
    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)
    '''
    sv = tf.train.Supervisor(logdir='hour', summary_op=None, init_fn=None, save_model_secs=100)
    gen = get_land(batch_size=batch_size, image_size=config.image_size)
    qq = data_loader_multi.get_thread(gen, 1)

    with sv.managed_session() as sess:
        ids = 0
        for step in range(100000000):
            t = time.time()
            org_im, msk = qq.get()
            fd = {image: org_im, mask: msk}
            ls = sess.run([train_op], feed_dict=fd)
            print(ls)
            if step % 100 == 0:
                out,  stp = sess.run([out_put, global_step], feed_dict=fd)

            if step % 100 == 0:
                for s in range(1):
                    plt.subplot(221)
                    plt.title('step1')
                    plt.imshow(msk[s, :, :, 0], aspect="auto", cmap='gray')

                    plt.subplot(222)
                    plt.title('step1')
                    plt.imshow(msk[s, :, :, 1], aspect="auto", cmap='gray')

                    plt.subplot(223)
                    plt.title('final')

                    plt.imshow(out[s, :, :, 0], aspect="auto", cmap='gray')
                    plt.subplot(224)
                    plt.title('final')
                    plt.imshow(out[s, :, :, 1], aspect="auto", cmap='gray')
                    plt.savefig('dd.jpg')



if __name__ == '__main__':
    run()
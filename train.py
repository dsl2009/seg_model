import tensorflow as tf
from loss import get_loss,sigmoid_cross_entropy_balanced,nn_loss
from matplotlib import pyplot as plt
from tensorflow.contrib import slim
from ai_chaellenger import get_drive
import config
import numpy as np
import time
from dsl_data import data_loader_multi
def run():
    batch_size = 8
    image = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 3))
    mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, 512, 512, 2))
    global_step = tf.train.get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate=0.002,
        global_step=global_step,
        decay_steps=100000,
        decay_rate=0.7,
        staircase=True)

    out_put, train_tensors, loss1, loss2 = get_loss(image, mask)
    out_put = tf.nn.sigmoid(out_put)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    vbs = []
    for s in slim.get_variables():
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name and 'GroupNorm' not in s.name:
            vbs.append(s)
    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)

    sv = tf.train.Supervisor(logdir='drive1', summary_op=None, init_fn=restore, save_model_secs=100)
    gen = get_drive(batch_size=batch_size, image_size=[512, 512])
    qq = data_loader_multi.get_thread(gen, 1)

    with sv.managed_session() as sess:
        ids = 0
        for step in range(100000000):
            t = time.time()
            org_im, msk = qq.get()
            fd = {image: org_im, mask: msk}
            ls = sess.run([train_op, loss1, loss2], feed_dict=fd)
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
                    print(out[s, :, :, 0])
                    print(np.sum(out))
                    plt.imshow(out[s, :, :, 0], aspect="auto", cmap='gray')
                    plt.subplot(224)
                    plt.title('final')
                    plt.imshow(out[s, :, :, 1], aspect="auto", cmap='gray')
                    plt.savefig('dd.jpg')

def eger():
    tf.enable_eager_execution()
    gen = get_drive(batch_size=4, image_size=[512, 512])
    org_im, msk = next(gen)
    print(sigmoid_cross_entropy_balanced(msk, msk))
    print(nn_loss(msk, msk))


if __name__ == '__main__':
    run()
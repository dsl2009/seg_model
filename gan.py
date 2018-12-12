from tensorflow.contrib import slim
import tensorflow as tf
import collections
from ai_chaellenger import get_land,get_tree
import config
import time
from matplotlib import pyplot as plt
from dsl_data import data_loader_multi
import os
import numpy as np
import glob
from skimage import io
from coor_conv import AddCoords
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake,"
                                        " discrim_loss, discrim_grads_and_vars, "
                                        "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, "
                                        "train")


def lrelu(x):
    a = 0.2
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
def seg_arg():
    batch_norm_params = {
        'is_training':True,
        'decay': 0.9,
        'epsilon':1e-5,
        'scale':True,
        'param_initializers' :{'gamma':tf.random_normal_initializer(1.0, 0.02)},
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.00001),
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.random_normal_initializer(0, 0.02),
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as sc:
            return sc

def create_generator(inputs):
    with tf.variable_scope("generator"):
        with slim.arg_scope(seg_arg()):
            x_128 = slim.conv2d(inputs,num_outputs=64,kernel_size=4,stride=2)
            x_64 = slim.conv2d(x_128, num_outputs=128, kernel_size=4, stride=2)
            x_32 = slim.conv2d(x_64, num_outputs=256, kernel_size=4, stride=2)
            x_16 = slim.conv2d(x_32, num_outputs=512, kernel_size=4, stride=2)
            x_8 = slim.conv2d(x_16, num_outputs=512, kernel_size=4, stride=2)
            x_4 = slim.conv2d(x_8, num_outputs=512, kernel_size=4, stride=2)
            x_2 = slim.conv2d(x_4, num_outputs=512, kernel_size=4, stride=2)
            x_1 = slim.conv2d(x_2, num_outputs=512, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            with slim.arg_scope([slim.conv2d],activation_fn=slim.relu):
                x = slim.conv2d_transpose(x_1, num_outputs=512, kernel_size=4, stride=2)
                #x = slim.dropout(x, keep_prob=0.9)
                x = slim.conv2d_transpose(tf.concat([x, x_2],axis=3), num_outputs=512, kernel_size=4, stride=2)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d_transpose(tf.concat([x, x_4],axis=3), num_outputs=512, kernel_size=4, stride=2)
                #x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d_transpose(tf.concat([x, x_8],axis=3), num_outputs=512, kernel_size=4, stride=2)
                x = slim.conv2d_transpose(tf.concat([x, x_16],axis=3), num_outputs=256, kernel_size=4, stride=2)
                x = slim.conv2d_transpose(tf.concat([x, x_32],axis=3), num_outputs=128, kernel_size=4, stride=2)
                x = slim.conv2d_transpose(tf.concat([x, x_64],axis=3), num_outputs=64, kernel_size=4, stride=2)
                x = slim.conv2d_transpose(tf.concat([x, x_128],axis=3), num_outputs=1, kernel_size=4, stride=2,
                                          normalizer_fn=None, activation_fn=None)
    return x, slim.nn.sigmoid(x)

def create_discriminator(inputs, target):
    with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
        with slim.arg_scope(seg_arg()):
            x = tf.concat([inputs, target], axis=3)
            x = slim.conv2d(x, num_outputs=64, kernel_size=4, stride=2)
            x = slim.conv2d(x, num_outputs=128, kernel_size=4, stride=2)
            x = slim.conv2d(x, num_outputs=256, kernel_size=4, stride=2)
            x = slim.conv2d(x, num_outputs=512, kernel_size=4, stride=1)
            x = slim.conv2d(x, num_outputs=1, kernel_size=4, stride=1, normalizer_fn=None, activation_fn=tf.nn.sigmoid)
        print(x)
    return x

def gen_model(inputs, targets,EPS =1e-12):
    logits, outputs = create_generator(inputs)
    predict_real = create_discriminator(inputs, targets)
    predict_fake = create_discriminator(inputs, outputs)

    discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
    sigmod_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

    gen_loss = gen_loss_GAN * 1 + sigmod_loss*100
    global_step = tf.train.get_or_create_global_step()

    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=100000,
        decay_rate=0.7,
        staircase=True)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, global_step,  gen_train),
    )



def run():
    batch_size = 16
    image = tf.placeholder(dtype=tf.float32, shape=(batch_size, config.image_size[0], config.image_size[1], 3))
    images = AddCoords(x_dim=256, y_dim=256)(image)
    mask = tf.placeholder(dtype=tf.float32, shape=(batch_size, config.image_size[0], config.image_size[1], 1))
    model = gen_model(inputs=images, targets=mask)

    sv = tf.train.Supervisor(logdir='land', summary_op=None, init_fn=None, save_model_secs=100)
    gen = get_land(batch_size=batch_size, image_size=config.image_size)
    qq = data_loader_multi.get_thread(gen, 1)

    with sv.managed_session() as sess:
        ids = 0
        for step in range(100000000):
            t = time.time()
            org_im, msk = qq.get()
            fd = {image: org_im, mask: msk}
            _ ,d_loss, g_loss, l1_loss = sess.run([model.train, model.discrim_loss, model.gen_loss_GAN, model.gen_loss_L1], feed_dict=fd)
            print(d_loss, g_loss, l1_loss)
            if step % 100 == 0:
                out = sess.run(model.outputs, feed_dict=fd)

            if step % 100 == 0:
                for s in range(1):
                    plt.subplot(221)
                    plt.title('step1')
                    plt.imshow(msk[s,:,:,0], aspect="auto")

                    plt.subplot(222)
                    plt.title('org')
                    plt.imshow(org_im[s]+np.asarray([123.15, 115.90, 103.06])/255.0, aspect="auto")

                    plt.subplot(223)
                    plt.title('pred')
                    plt.imshow(out[s,:,:,0], aspect="auto")

                    plt.subplot(224)
                    plt.title('org')
                    plt.imshow(org_im[s] + np.asarray([123.15, 115.90, 103.06]) / 255.0, aspect="auto")
                    plt.savefig('dd.jpg')

def detect():
    batch_size = 1
    image = tf.placeholder(dtype=tf.float32, shape=(1, config.image_size[0], config.image_size[1], 3))
    images = AddCoords(x_dim=256, y_dim=256)(image)
    logits, out_put = create_generator(images)
    sv = tf.train.Supervisor(logdir='land', summary_op=None, init_fn=None, save_model_secs=100)
    gen = get_land(batch_size=batch_size, image_size=config.image_size)
    qq = data_loader_multi.get_thread(gen, 1)

    with sv.managed_session() as sess:
        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/land/*.png'):
            org_im, msk = qq.get()
            fd = {image: org_im}
            msked = sess.run(out_put, fd)
            plt.subplot(211)
            plt.title('step1')
            plt.imshow(msk[0,:,:,0])
            plt.subplot(212)
            plt.title('step1')
            plt.imshow(msked[0,:,:,0])
            plt.show()




if __name__ == '__main__':
    detect()
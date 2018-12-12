from tensorflow.contrib import slim
import tensorflow as tf
from nets import resnet_v2,resnet_utils
from loss import sigmoid_cross_entropy_balanced1,sec_losses

def resnet_v2_block(inputs,scope='block1', in_depth=64,add=None,is_downsample=False,rate=2):
    with tf.variable_scope(scope):
        orig_x = inputs
        x = slim.group_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if add is not None:
            x = add
        with tf.variable_scope('sub1'):
            if is_downsample:
                x = slim.conv2d(x,num_outputs=in_depth,kernel_size=1,stride=2)
            else:
                x = slim.conv2d(x, num_outputs=in_depth, kernel_size=1, stride=1)

        with tf.variable_scope('sub2'):
            x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1,rate=rate)

        with tf.variable_scope('sub3'):
            x = slim.conv2d(x, num_outputs=in_depth*4, kernel_size=1, stride=1)

        with tf.variable_scope('sub_add'):

             x += orig_x


    return x



def identity_block(input_tensor, in_depth,rate,scope):
    with tf.variable_scope(scope):
        x = slim.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=1)
        x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
        x = slim.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1,activation_fn=None)
        x += input_tensor
        return tf.nn.relu(x)



def conv_block(input_tensor, in_depth,  rate,stride=2):
    x = slim.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=stride)
    x = slim.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
    x = slim.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1, activation_fn=None)
    shortcut =  x = slim.conv2d(input_tensor, num_outputs=in_depth* 4, kernel_size=1, stride=stride,activation_fn=None)
    x +=shortcut
    return tf.nn.relu(x)

def seg_arg_gn():
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.00001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.group_norm,
            padding = 'SAME') as sc:
        return sc




image = tf.placeholder(dtype=tf.float32,shape=(1,256,256,3))
num_label =1

def create_model(inputs):
    with slim.arg_scope(seg_arg_gn()):
        conv1 = slim.conv2d(inputs, num_outputs=64, kernel_size=7, stride=1)
        pol1 = slim.max_pool2d(conv1, kernel_size=3, stride=2,padding='SAME')

        # Stage 2
        conv2 = conv_block(pol1, 64, rate=2, stride=1)
        conv2 = slim.repeat(conv2, 2, identity_block, scope='block1',in_depth=64, rate=2)
        # Stage 3
        conv3 = conv_block(conv2, 128, rate=2, stride=2)
        conv3 = slim.repeat(conv3, 3, identity_block, scope='block2',in_depth=128, rate=2)
        # Stage 4
        conv4 = conv_block(conv3, 128, rate=2, stride=2)
        conv4 = slim.repeat(conv4, 3, identity_block, scope='block3',in_depth=128, rate=2)
        # Stage 5
        conv5 = conv_block(conv4, 128, rate=2, stride=2)
        conv5 = slim.repeat(conv5, 2, identity_block,scope='block4', in_depth=128, rate=2)

        # Stage 6
        conv6 = conv_block(conv5, 128, rate=2, stride=2)
        conv6 = slim.repeat(conv6, 2, identity_block, scope='block5', in_depth=128, rate=2)

        # Stage 7
        conv7 = conv_block(conv6, 128, rate=2, stride=2)
        conv7 = slim.repeat(conv7, 2, identity_block, scope='block6', in_depth=128, rate=2)



        print(conv7)

create_model(image)





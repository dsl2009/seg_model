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


def bliliner_additive_upsampleing(featear,out_channel,stride):

    in_channel = featear.get_shape().as_list()[3]
    assert in_channel % out_channel == 0
    channel_split = in_channel/out_channel
    channel_split = tf.cast(channel_split, tf.int32)
    new_shape = featear.get_shape().as_list()
    new_shape[1] *= stride
    new_shape[2] *= stride
    new_shape[3] *= out_channel
    up_sample_feature = tf.image.resize_bilinear(featear,new_shape[1:3])
    out_list = []
    for i in range(out_channel):
        splited_upsample = up_sample_feature[:,:,:,i*channel_split:(i+1)*channel_split]
        out_list.append(tf.reduce_sum(splited_upsample,axis=-1))
    fea = tf.stack(out_list,axis=-1)
    fea = slim.separable_conv2d(fea,out_channel,kernel_size=stride*2,depth_multiplier=4,activation_fn=tf.nn.tanh)
    return fea


image = tf.placeholder(dtype=tf.float32,shape=(1,256,256,3))
num_label =1

def create_model(inputs, labels=None,cts= None,is_train=True):
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
        conv4 = conv_block(conv3, 256, rate=2, stride=2)
        conv4 = slim.repeat(conv4, 22, identity_block, scope='block3',in_depth=256, rate=2)
        # Stage 5
        conv5 = conv_block(conv4, 512, rate=2, stride=2)
        conv5 = slim.repeat(conv5, 2, identity_block,scope='block4', in_depth=512, rate=4)


        cv2_1 = slim.conv2d(conv2, num_outputs=256, kernel_size=1, stride=1)
        dev2 = slim.conv2d_transpose(cv2_1, num_outputs=num_label, kernel_size=3, stride=2, normalizer_fn=None, activation_fn=tf.nn.tanh)

        cv3_1 = slim.conv2d(conv3, num_outputs=256, kernel_size=1, stride=1)
        dev3 = slim.conv2d_transpose(cv3_1, num_outputs=num_label, kernel_size=3, stride=4, normalizer_fn=None, activation_fn=tf.nn.tanh)

        cv4_1 = slim.conv2d(conv4, num_outputs=256, kernel_size=1, stride=1)
        dev4 = slim.conv2d_transpose(cv4_1, num_outputs=num_label, kernel_size=3, stride=8, normalizer_fn=None, activation_fn=tf.nn.tanh)

        cv5_1 = slim.conv2d(conv5, num_outputs=256, kernel_size=1, stride=1)
        dev5 = slim.conv2d_transpose(cv5_1, num_outputs=num_label, kernel_size=3, stride=16, normalizer_fn=None, activation_fn=tf.nn.tanh)

        contt = tf.concat([dev2, dev3, dev4, dev5], 3)

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(0.0001),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                ):
                x = slim.conv2d(contt, num_outputs=num_label, kernel_size=1, stride=1,normalizer_fn=None,
                                 weights_initializer=tf.constant_initializer(0.2), activation_fn=tf.nn.tanh)
        final_loss = sigmoid_cross_entropy_balanced1(logits=x,labels=labels)
        dev2_loss = sigmoid_cross_entropy_balanced1(logits=dev2,labels=labels)
        dev3_loss = sigmoid_cross_entropy_balanced1(logits=dev3, labels=labels)
        dev4_loss = sigmoid_cross_entropy_balanced1(logits=dev4, labels=labels)
        dev5_loss = sigmoid_cross_entropy_balanced1(logits=dev5, labels=labels)

        total_loss = final_loss+dev2_loss*0.3+dev3_loss*0.4+dev4_loss*0.5+dev5_loss*0.6
        tf.losses.add_loss(total_loss)
        total_loss = tf.losses.get_total_loss()
        train_tensors = tf.identity(total_loss, 'ss')
        return train_tensors, x







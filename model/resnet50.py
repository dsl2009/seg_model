from nets.resnet_v2 import resnet_v2_block,resnet_v2,resnet_arg_scope
import tensorflow as tf
from tensorflow.contrib import slim
import config

def resnet_arg_scope_batch_norm(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def resnet_arg_scope_group_norm(weight_decay=0.0001,
                                activation_fn=tf.nn.relu,
                               ):
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.group_norm,
            ):
        with slim.arg_scope([slim.group_norm]):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
def seg_arg():
    batch_norm_params = {
        'is_training':True,
        'decay': 0.9997,
        'epsilon':1e-5,
        'scale':True
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.00001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as sc:
            return sc



def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  print(inputs)
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
if config.is_use_group_norm:
    base_arg = resnet_arg_scope_group_norm
else:
    base_arg = resnet_arg_scope

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
    print(up_sample_feature)
    out_list = []
    for i in range(out_channel):
        splited_upsample = up_sample_feature[:,:,:,i*channel_split:(i+1)*channel_split]
        out_list.append(tf.reduce_sum(splited_upsample,axis=-1))
    fea = tf.stack(out_list,axis=-1)
    #fea = slim.conv2d(fea,out_channel,kernel_size=stride*2,activation_fn=tf.nn.tanh)
    fea = slim.separable_conv2d(fea,out_channel,kernel_size=stride*2,depth_multiplier=4,activation_fn=tf.nn.tanh)
    return fea


def fpn1(img):
    with slim.arg_scope(base_arg()):
        _, endpoint = resnet_v2_50(img)
    c1 = endpoint['resnet_v2_50/block1']
    c2 = endpoint['resnet_v2_50/block2']
    c3 = endpoint['resnet_v2_50/block3']
    c4 = endpoint['resnet_v2_50/block4']
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=slim.xavier_initializer(),
                        activation_fn=tf.nn.relu,
                        biases_initializer= slim.init_ops.constant_initializer(0.2)
                        ):
        c3_1 = slim.conv2d(c3, num_outputs=256, kernel_size=1, rate=1)
        c3_2 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=6)
        c3_3 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=12)
        c3_4 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=18)
        c3 = tf.concat([c3_1,c3_2,c3_3,c3_4],axis=3)
        c3 = slim.conv2d(c3, num_outputs=256, kernel_size=1)
        c1 = slim.conv2d(c1, num_outputs=256, kernel_size=1)
        c3 = tf.image.resize_bilinear(c3, tf.shape(c1)[1:3])
        x = tf.concat([c1,c3],axis=3)
        x = slim.conv2d(x, num_outputs=2, kernel_size=3, activation_fn=tf.nn.tanh)
        x = tf.image.resize_bilinear(x, tf.shape(img)[1:3])
    return x

def fpn(img):
    with slim.arg_scope(base_arg()):
        _, endpoint = resnet_v2_50(img)
    c1 = endpoint['resnet_v2_50/block1']
    c2 = endpoint['resnet_v2_50/block2']
    c3 = endpoint['resnet_v2_50/block3']
    c4 = endpoint['resnet_v2_50/block4']
    with slim.arg_scope(seg_arg()):
        c3_1 = slim.conv2d(c3, num_outputs=256, kernel_size=1, rate=1)
        c3_2 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=6)
        c3_3 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=12)
        c3_4 = slim.conv2d(c3, num_outputs=256, kernel_size=3, rate=18)
        c31 = tf.concat([c3_1, c3_2, c3_3, c3_4], axis=3)
        c31 = slim.conv2d(c3, num_outputs=256, kernel_size=1)
        c31 = tf.image.resize_bilinear(c3, tf.shape(c1)[1:3])

        c4_1 = slim.conv2d(c4, num_outputs=256, kernel_size=1, rate=1)
        c4_2 = slim.conv2d(c4, num_outputs=256, kernel_size=3, rate=6)
        c4_3 = slim.conv2d(c4, num_outputs=256, kernel_size=3, rate=12)
        c4_4 = slim.conv2d(c4, num_outputs=256, kernel_size=3, rate=18)
        c41 = tf.concat([c4_1, c4_2, c4_3, c4_4], axis=3)
        c41 = slim.conv2d(c4, num_outputs=256, kernel_size=1)
        c41 = tf.image.resize_bilinear(c4, tf.shape(c1)[1:3])

        c2_1 = slim.conv2d(c2, num_outputs=256, kernel_size=1, rate=1)
        c2_2 = slim.conv2d(c2, num_outputs=256, kernel_size=3, rate=6)
        c2_3 = slim.conv2d(c2, num_outputs=256, kernel_size=3, rate=12)
        c2_4 = slim.conv2d(c2, num_outputs=256, kernel_size=3, rate=18)
        c21 = tf.concat([c2_1, c2_2, c2_3, c2_4], axis=3)
        c21 = slim.conv2d(c2, num_outputs=256, kernel_size=1)
        c21 = tf.image.resize_bilinear(c2, tf.shape(c1)[1:3])

        c11 = slim.conv2d(c1, num_outputs=256, kernel_size=1)

        x1 = tf.concat([c11,c21,c31, c41], axis=3)
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(0.0001),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                ):
                x = slim.conv2d(x1, num_outputs=2, kernel_size=3, normalizer_fn=None, activation_fn=None)
                x = tf.image.resize_bilinear(x, tf.shape(img)[1:3])
    return x,x1


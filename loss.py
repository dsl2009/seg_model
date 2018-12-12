from model import resnet50
import tensorflow as tf
from model import hourglas_net
def sigmoid_cross_entropy_balanced(logits, labels, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """

    y = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


def sigmoid_cross_entropy_balanced1(logits, labels, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """

    y = tf.cast(labels, tf.float32)


    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost)

    # check if image has no edge pixels return 0 else return complete error function
    return cost

def nn_loss(logits, labels, name='cross_entropy_loss'):
    labels = tf.cast(labels, tf.float32)

    logits = tf.cast(logits, tf.float32)

    t_loss = tf.keras.backend.switch(tf.cast(tf.size(labels) > 0, tf.bool),
                                     tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                                     tf.constant(0.0))
    t_loss = tf.reduce_mean(t_loss)
    return t_loss

def sec_loss(result, labels):
    result = tf.nn.sigmoid(result)
    y = tf.cast(labels, tf.float32)
    count_pos = tf.reduce_sum(y)
    mask1 = result[:, :, :, 0]
    mask2 = result[:, :, :, 1]
    los = tf.reduce_sum(mask1*mask2)/count_pos
    los = tf.sqrt(los)
    return los

def sec_losses(result, labels, num_labels):
    result = tf.nn.sigmoid(result)
    y = tf.cast(labels, tf.float32)
    count_pos = tf.reduce_sum(y)
    los = []
    for i in range(num_labels):
        for j in range(i+1, num_labels):
            mask1 = result[:, :, :, i]
            mask2 = result[:, :, :, j]
            los.append(tf.reduce_sum(mask1 * mask2))
    los = tf.reduce_sum(los)/count_pos
    los = tf.sqrt(los)
    return los


def get_loss(img,labels):
    #result = resnet50.fpn(img)
    result = hourglas_net.fcn(img)
    ls = sigmoid_cross_entropy_balanced1(result, labels)
    se_loss = sec_loss(result, labels)
    tf.losses.add_loss(ls)
    tf.losses.add_loss(se_loss)
    total_loss = tf.losses.get_total_loss()

    tf.summary.scalar(name='loss', tensor=ls)
    train_tensors = tf.identity(total_loss, 'ss')
    return result, train_tensors, ls, se_loss



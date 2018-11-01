import tensorflow as tf

def fidelity(output, target):

    fidelity =tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target))))
    return fidelity


def tracedistance(output, target):
    tracedistance = tf.reduce_mean(tf.sqrt(1 - tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target)))))
    return tracedistance


def msq(output,target):
    #return tf.losses.mean_squared_error(target,output)
    return tf.reduce_mean(tf.square(tf.abs(output-target)))
    #return tf.reduce_sum(tf.square(tf.abs(output - target)))


def msq_fq(output,target,scale=0.5):
    return scale * fidelity(output,target)-(1-scale)*msq(output, target)

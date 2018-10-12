import tensorflow as tf

def fidelity(output, target):

    fidelity =tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target))))
    return fidelity


def tracedistance(output, target):
    tracedistance = tf.sqrt(1 - tf.square(fidelity(output, target)))
    return tracedistance



def fidelity_batch(output, target):
    c = lambda i,x,y: tf.less(i, target.shape[1])
    b = lambda i,x,y:(fidelity(x[i],y[i]),i + 1,)
    test = tf.while_loop(c, b, [0,output,target])

    return test



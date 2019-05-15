import tensorflow as tf
import unionlayer

def fidelity(output, target):

    fidelity =tf.reduce_mean(tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target))))
    return fidelity

def fidelity_list(output, target):

    fidelity_l =tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target)))
    return fidelity_l

def fidelity_partial(output, target,nbqubitout,nbqutbittarget):
    #print(output)
    #print(target)
    partial=tf.ones([2**(nbqubitout-nbqutbittarget),tf.shape(output)[1]],dtype=output.dtype)
    #partialnorm=tf.div(partial, tf.norm(partial, axis=0))
    target2=unionlayer.join(partial,target)
    #print(target2)
    temp=tf.abs(tf.reduce_sum(tf.square(tf.einsum('ij,ij->ij', tf.conj(target2), output)), axis=0))
    #print(temp)
    return tf.reduce_mean(temp)

def fidelity_partial_list(output, target,nbqubitout,nbqutbittarget):
    #print(output)
    #print(target)
    partial=tf.ones([2**(nbqubitout-nbqutbittarget),tf.shape(output)[1]],dtype=output.dtype)
    #partialnorm=tf.div(partial, tf.norm(partial, axis=0))
    target2=unionlayer.join(partial,target)
    #print(target2)
    temp=tf.abs(tf.reduce_sum(tf.square(tf.einsum('ij,ij->ij', tf.conj(target2), output)), axis=0))
    #print(temp)
    return temp


def majority_metric(fd_list):
    temp = tf.where(fd_list > 0.50, tf.ones_like(fd_list),tf.zeros_like(fd_list))
    return tf.reduce_mean(temp)

def maxclass(output, labels,nbqubitout,nbqutbittarget,nbclass, encoding):
    # la liste est inefficace
    problist =[]

    for x in range(0,nbclass):
        if encoding == 'inttoqubit' :
            temptarget =  unionlayer.inttoqubit(tf.fill(tf.shape(labels),x),nbqutbittarget)

        elif encoding == 'onehot':
            temptarget = unionlayer.onehotqubits(tf.fill(tf.shape(labels),x), nbqutbittarget)

        problist.append(fidelity_partial_list(output,temptarget,nbqubitout,nbqutbittarget))


    mlist=tf.argmax(tf.stack(problist, 1), 1)

    return mlist #tf.where(mlist == tf.cast(label,dtype="int32"), tf.ones_like(fd_list),tf.zeros_like(fd_list))

def max_metric(output, labels,nbqubitout,nbqutbittarget,nbclass, encoding):
    temp = maxclass(output, labels,nbqubitout,nbqutbittarget,nbclass, encoding)
    return tf.reduce_mean(tf.where(tf.equal(temp,tf.cast(labels,dtype='int64')),tf.ones_like(labels),
                                   tf.zeros_like(labels)))

def cross_entropy(output, target,nbqubitout,nbqutbittarget):
    fd_list=fidelity_partial_list(output,target,nbqubitout,nbqutbittarget)
    log_fd= tf.log(fd_list)
    return tf.reduce_mean(-1 * log_fd)


def tracedistance(output, target):
    tracedistance = tf.reduce_mean(tf.sqrt(1 - tf.square(tf.abs(tf.einsum('ij,ij->j', tf.conj(output), target)))))
    return tracedistance


def msq(output,target):
    #return tf.losses.mean_squared_error(target,output)
    return output.shape[1]*tf.reduce_mean((tf.square(tf.abs(output-target))))
    #return tf.reduce_sum(tf.square(tf.abs(output - target)))

def msq_real(output,target):
    return tf.cast(tf.shape(target)[0],dtype=target.dtype)*tf.losses.mean_squared_error(target,output)

def msq_fq(output,target,scale=0.5):
    return scale * fidelity(output,target)-(1-scale)*msq(output, target)


if __name__ == "__main__" :
    tf.enable_eager_execution()
    real = [[1 / tf.sqrt(2.0),0.0,1 / tf.sqrt(2.0),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[1 / tf.sqrt(2.0),1 / tf.sqrt(2.0),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    imag = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    test = tf.transpose(tf.complex(real, imag))
    real = [[1 / tf.sqrt(2.0), 0.0,1 / tf.sqrt(2.0),0.0 ], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0),0.0, 0.0]]
    imag = [[0.0, 0.0,0.0, 0.0], [0.0, 0.0,0.0, 0.0]]
    un =  tf.transpose(tf.complex(real, imag))
    real = [[1.0, 0.0],[1.0 ,0.0]]
    imag = [[0.0, 0.0],[0.0 ,0.0]]
    zero = tf.transpose(tf.complex(real, imag))

    print(fidelity_partial_list(un,zero,2,1))
    fd_list=fidelity_partial_list(un, zero, 2, 1)
    print(majority_metric(fd_list))
    print(cross_entropy(un, zero,2,1))
    label=tf.transpose(tf.cast([0.0,1.0], dtype="float32"))
    print(maxclass(un, label, 2, 1, 2, 'inttoqubit'))
    print(max_metric(un,label,2,1,2,'inttoqubit'))

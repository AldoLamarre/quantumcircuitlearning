import tensorflow as tf

def join(inputleft,inputright):

    return tf.reshape(tf.einsum('ib,jb->ijb',inputleft,inputright),[tf.shape(inputleft)[0]*tf.shape(inputright)[0],tf.shape(inputleft)[1]])

def inttoqubit(input,nbqubit):
    #input= tf.squeeze(input)
    size = 1 << nbqubit
    row=tf.range(0,tf.cast(tf.shape(input)[0],dtype="int64"),dtype="int64")
    pos=tf.stack([row,tf.squeeze(tf.to_int64(input))],axis=1)
    ones=tf.ones_like(tf.squeeze(input),dtype="float32")
    vec=tf.sparse_tensor_to_dense(tf.SparseTensor(pos,ones,[tf.shape(input)[0],size]))
    imag=tf.zeros_like(vec,dtype="float32")
    return tf.transpose(tf.complex(vec,imag))



def onehotqubits(input,nbqubit):
    size = 1 << nbqubit
    row=tf.range(0,tf.cast(tf.shape(input)[0],dtype="int64"),dtype="int64")
    pos=tf.stack([row,tf.squeeze(tf.to_int64(2**input))],axis=1)
    ones=tf.ones_like(tf.squeeze(input),dtype="float32")
    vec=tf.sparse_tensor_to_dense(tf.SparseTensor(pos,ones,[tf.shape(input)[0],size]))
    imag=tf.zeros_like(vec,dtype="float32")
    return tf.transpose(tf.complex(vec,imag))


if __name__ == "__main__":
    real = [[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    un = tf.complex(real, imag)
    real = [[1.0, 0.0],[0.0 ,1.0]]
    imag = [[0.0, 0.0],[0.0 ,0.0]]
    zero = tf.complex(real, imag)
    r=join(zero,un)

    t=onehotqubits(tf.constant([0,1]),2)

    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        sess.run(start)
        r1,t1 = sess.run([r,t])
        print(r1)
        print(t1)

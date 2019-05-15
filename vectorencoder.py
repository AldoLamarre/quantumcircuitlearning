import tensorflow as tf

# arxiv 1804.00633



def encode_vector_one(vector, nbqubit, constant_procedure ):
    #if tf.size(vector) > 2** nbqubit:
        # handle error correctly
        #return vector

    ctes = constant_procedure(vector,nbqubit)

    unnormvector= tf.pad(vector,[[0,2** nbqubit-tf.size(vector)]],"constant_values",constant_values=ctes)

    normvertor=tf.div(unnormvector,tf.norm(unnormvector,axis =0))

    return normvertor


def encode_vectors(vectors, nbqubit, ctes):

    #if tf.shape(vectors)[1] > 2 ** nbqubit:
        # handle error correctly
        #return vectors


    unnormvector = tf.transpose(tf.concat([vectors,ctes],axis=1))

    normvertor = tf.div(unnormvector, tf.norm(unnormvector, axis=0))
    cplx=tf.zeros_like(normvertor)

    return tf.transpose(tf.complex(normvertor,cplx))


def gennormpluscte(vectors, nbqubit,cte=0.01):
    norms=tf.norm(vectors, axis=1)
    #verify shape to handle error maybe
    #[(2 ** nbqubit)-tf.shape(vectors)[0],tf.shape(vectors)[1]]
    #print(vectors.dtype)
    #print(shape)
    ctes=tf.fill([tf.shape(vectors)[0],(2 ** nbqubit)-tf.shape(vectors)[1]-1],cte)
    #print(ctes)
    temp=tf.concat([tf.expand_dims(norms,1),ctes],axis=1)
    return temp

def gencte(vectors, nbqubit,cte=0.01):

    #verify shape to handle error maybe
    #[(2 ** nbqubit)-tf.shape(vectors)[0],tf.shape(vectors)[1]]

    #print(shape)
    ctes=tf.fill([tf.shape(vectors)[0],(2 ** nbqubit)-tf.shape(vectors)[1]],cte)


    return ctes


if __name__ == "__main__":
    #tf.enable_eager_execution()
    data=tf.Variable([[0.0,0.0,2.0],[0.0,1.0,1.0],[1.0,0.0,0.0]])
    #print(data.shape)

    ctes=gennormpluscte(data,4,cte=0.05)

    encodedvectors=encode_vectors(data,4,ctes)
    #print(encodedvectors)
    norm=tf.norm(encodedvectors, axis=1)
    init = tf.orthogonal_initializer
    # real = tf.get_variable("v", [2**8, 2**8], initializer=init)
    real = tf.eye(2 ** 4, dtype="float32")
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    result= tf.matmul(param,tf.transpose(encodedvectors))
    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        sess.run(start)
        r1,n,r = sess.run([encodedvectors,norm,result])
        print(r1)
        print(n)
        print(r)



import tensorflow as tf

#To debug
def lazymeasure(input):

    split0,split1=tf.split(tf.convert_to_tensor (input), 2,axis=0)

    prob0,prob1=tf.real(tf.square(tf.norm(split0,axis=0))), tf.real(tf.square(tf.norm(split1, axis=0)))
    temp=tf.random_uniform(tf.shape(prob0), 0, 1)
    #temp=tf.zeros_like(prob0)
    #temp = tf.where(prob0==0,temp,tf.ones_like(prob0))
    #temp = tf.where(prob1==0,temp, tf.zeros_like(prob1))
    remain=tf.transpose(tf.where(temp<=prob0,tf.transpose(split0),tf.transpose(split1)))
    result=tf.where(temp<=prob0,tf.zeros_like(prob0),tf.ones_like(prob0))



    return tf.div(remain,tf.norm(remain,axis=0)),result


def lazymeasureone(input):

    split0,split1=tf.split(tf.convert_to_tensor (input), 2,axis=0)

    prob0,prob1=tf.real(tf.square(tf.norm(split0,axis=0))), tf.real(tf.square(tf.norm(split1, axis=0)))
    #temp=tf.random_uniform(tf.shape(prob0), 0, 1)
    temp=tf.ones_like(prob0)
    #temp = tf.where(prob0==0,temp,tf.ones_like(prob0))
    #temp = tf.where(prob1==0,temp, tf.zeros_like(prob1))
    remain=tf.transpose(tf.where(temp<=prob0,tf.transpose(split0),tf.transpose(split1)))
    result=tf.where(temp<=prob0,tf.zeros_like(prob0),tf.ones_like(prob0))



    return tf.div(remain,tf.norm(remain,axis=0)),result


def lazymeasurezero(input):

    split0,split1=tf.split(tf.convert_to_tensor (input), 2,axis=0)

    prob0,prob1=tf.real(tf.square(tf.norm(split0,axis=0))), tf.real(tf.square(tf.norm(split1, axis=0)))
    #temp=tf.random_uniform(tf.shape(prob0), 0, 1)
    temp=tf.zeros_like(prob0)
    #temp = tf.where(prob0==0,temp,tf.ones_like(prob0))
    #temp = tf.where(prob1==0,temp, tf.zeros_like(prob1))
    remain=tf.transpose(tf.where(temp<=prob0,tf.transpose(split0),tf.transpose(split1)))
    result=tf.where(temp<=prob0,tf.zeros_like(prob0),tf.ones_like(prob0))



    return tf.div(remain,tf.norm(remain,axis=0)),result


def lazymeasureboth(input):
    return lazymeasurezero(input),lazymeasureone(input)

def corelatedlazymeasure(input,coor):


    split0, split1 = tf.split(tf.convert_to_tensor(input), 2, axis=0)
    coorsplit0, coorsplit1 = tf.split(tf.convert_to_tensor(coor), 2, axis=0)

    prob0, prob1 = tf.real(tf.square(tf.norm(split0, axis=0))), tf.real(tf.square(tf.norm(split1, axis=0)))
    coorprob0, coorprob1 = tf.real(tf.square(tf.norm(coorsplit0, axis=0))), tf.real(tf.square(tf.norm(coorsplit0, axis=0)))

    temp = tf.random_uniform(prob0.shape, 0, 1)

    # temp = tf.where(prob0==0,temp,tf.ones_like(prob0))
    # temp = tf.where(prob1==0,temp, tf.zeros_like(prob1))
    remain = tf.transpose(tf.where(temp <= prob0, tf.transpose(split0), tf.transpose(split1)))
    result = tf.where(temp <= prob0, tf.zeros_like(prob0), tf.ones_like(prob0))
    coorresult = tf.where(temp <= coorprob0, tf.zeros_like(coorprob0), tf.ones_like(coorprob1))
    coorremain = tf.transpose(tf.where(temp <= coorprob0, tf.transpose(coorsplit0), tf.transpose(coorsplit1)))


    return tf.div(remain,tf.norm(remain,axis=0)),result,tf.div(coorremain,tf.norm(coorremain,axis=0)),coorresult

if __name__ == "__main__":
    real = [[0.5 ,0.5,0.5,0.5],[1 / tf.sqrt(2.0),0.0,0.0, 0.0]]
    imag = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,-1 / tf.sqrt(2.0)]]
    plus = tf.complex(real, imag)

    m=corelatedlazymeasure(tf.transpose(plus),tf.transpose(plus))
    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        sess.run(start)
        m1=sess.run([m])
        print(m1)

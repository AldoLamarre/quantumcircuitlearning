import tensorflow as tf
import lossfunction
import genericQGate
import lazymeasure
import datetime
import time
#from tensorflow.python import debug as tf_debug


#tf.enable_eager_execution()

if __name__ == "__main__":

    real = [[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    hadamard = tf.complex(real, imag)
    identity = tf.complex([[1.0, 0.0], [0.0, 1.0]],imag)
    identity2 = tf.complex([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],
                           [[0.0, 0.0,0.0,0.0], [0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])

    real = [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]]
    imag = [[0.0, 0.0,0.0,0.0], [0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    cnot = tf.complex(real, imag)

    with tf.variable_scope("gateh"):
        gateh = genericQGate.genericQGate(hadamard, 2, 1, 0, learningrate=1)
    with tf.variable_scope("gatecnot"):
        gatecnot = genericQGate.genericQGate(cnot, 3, 2, 1, learningrate=1)
    with tf.variable_scope("gate2"):
        gate2 = genericQGate.genericQGate(identity2, 3, 2, 1, learningrate=0.001,momentum=0.01)



    #init= tf.initializers.random_uniform(0, 1)
    prev=tf.zeros_like(gate2.param)

    real = tf.random_normal([8,256],0,1)
    imag = tf.random_normal([8,256],0,1)
    #real = [1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)]
    #imag = [0.0, 0.0]


    temp = tf.complex(real, imag)
    vectorrand= tf.div(temp,tf.norm(temp,axis =0))
    #print(vectorrand)
    #test1=tf.norm(vectorrand,axis =0)


    vector4,rv,target,rt = lazymeasure.corelatedlazymeasure(gate2.forward(vectorrand),gatecnot.forward(vectorrand))
    cost = lossfunction.fidelity(vector4,target)
    #dW = tf.squeeze(tf.gradients(ys=cost, xs=gate2.param))
    #dW += 0.01 * prev
    #prev = dW
    #update=gate2.update(dW)
    update=gate2.sgd(cost)
    test=tf.matmul(gate2.param, gate2.param, adjoint_a=True)
    flag=1
    max=0
    itermax=0
    f = open("log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")

    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        sess.run(start)
        for x in range(0, 25000):
            print("iter:" + str(x))
            c,g,t=sess.run([cost,update,test])
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            print(c)
            print(g)
            #print((gate2.param.eval()))
            f.write("iter:" + str(x)+"\n"+str(c)+"\n"+str(g)+"\n"+"iter end\n")
            #print(t)

            print("iter end\n")
            # if(c>0.95 and flag==1):
            #     flag=0
            #     #gate2.learningrate/=10
            # if (c > 0.99 and flag == 0):
            #     flag = 2
            #     gate2.learningrate /= 10
            # if (c > 0.995 and flag == 2):
            #     flag = 3
            #     gate2.learningrate /= 10
            #
            # if (c < 0.98 and flag >= 2):
            #    break


            if (c > max):
                max=c
                itermax=x
                print(g)





            #print(gateh.param.eval())
        print(max)
        print(itermax)
        f.write("\n"+str(max))
        f.write("\n"+str(itermax))



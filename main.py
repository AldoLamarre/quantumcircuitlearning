import tensorflow as tf
import lossfunction
import genericQGate
import lazymeasure
import unionlayer
import datetime
import time
import subcircuit
#from tensorflow.python import debug as tf_debug


#tf.enable_eager_execution()

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


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
    hitoricalfd=0
    hitoricalmsq = 0


    init = tf.orthogonal_initializer
    real = tf.get_variable("v", [2 ** 2, 2 ** 2], initializer=init)
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    with tf.variable_scope("gateh"):
        gateh = genericQGate.genericQGate(hadamard, 2, 1, 0, learningrate=1)
    with tf.variable_scope("gatecnot"):
        gatecnot = genericQGate.genericQGate(cnot, 3, 2, 0, learningrate=1)
    with tf.variable_scope("gate2"):
        gate2 = genericQGate.genericQGate(identity2, 3, 2, 0, learningrate=0.0001,momentum=0.01)

    #cir = subcircuit.ArityFillCircuit(2, 2, 1, "test0", 0.001, 0.01)



    #init= tf.initializers.random_uniform(0, 1)
    #prev=tf.zeros_like(gate2.param)

    real = tf.random_normal([8,1024],0,1)
    imag = tf.random_normal([8,1024],0,1)
    #real = [1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)]
    #imag = [0.0, 0.0]


    temp = tf.complex(real, imag)
    vectorrand= tf.div(temp,tf.norm(temp,axis =0))
    #print(vectorrand)
    #test1=tf.norm(vectorrand,axis =0)


    #vector4,rv,tem,rt = lazymeasure.corelatedlazymeasure(gate2.forward(vectorrand),gatecnot.forward(vectorrand))
    #out=unionlayer.join(unionlayer.inttoqubit(rv,1),vector4)
    #target=unionlayer.join(unionlayer.inttoqubit(rt,1),tem)
    out=gate2.forward(vectorrand)
    target=gatecnot.forward(vectorrand)
    cost = lossfunction.fidelity(out,target)
    trace = lossfunction.tracedistance(out, target)
    costmq = lossfunction.msq_fq(out, target)
    costm = lossfunction.msq(out, target)
    #norm = tf.reduce_sum([tf.square(tf.abs(tf.norm(w.param,axis = [-2, -1])))for w in cir.gatelist])
    #dW = tf.squeeze(tf.gradients(ys=cost, xs=gate2.param))
    #dW += 0.01 * prev
    #prev = dW
    #update=gate2.update(dW)
    scost=tf.summary.scalar(name='cost', tensor=cost)
    scostm = tf.summary.scalar(name='costm', tensor=costm)

    update=gate2.sgd(cost)
    supdate=tf.summary.tensor_summary(name="param",tensor=gate2.param)
    #gatelist=cir.gatelist
    #test=tf.matmul(gate2.param, gate2.param, adjoint_a=True)
    flag=1
    max=0
    itermax=0
    summaries = tf.summary.merge_all()
    f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        sess.run(start)
        writer=tf.summary.FileWriter("board\\log_" + time.strftime("%Y%m%d-%H%M%S"))


        for x in range(0, 500000):

            print("iter:" + str(x))
            c,cm,g,s=sess.run([cost,costm,update,summaries])
            #l=sess.run(gatelist)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            print(c)
            print(cm)
            #print(n)
            #print(g)
            #print((gate2.param.eval()))
           # f.write("iter:" + str(x)+"\n"+str(c)+"\n"+str(g)+"\n"+"iter end\n")
            writer.add_summary(s, x)
            #writer.add_summary(cm, x)

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
                f.write("iter:" + str(x) + "\n" + str(c) + "\n" + str(g) + "\n" + "iter end\n")
                #f.write(l)
                #print(g)





            #print(gateh.param.eval())
        print(max)
        print(itermax)
       #f.write("\n"+str(max))
        #f.write("\n"+str(itermax))
        #f.write("\n" +time.strftime("%Y%m%d-%H%M%S"))




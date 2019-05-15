import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction

import time



if __name__ == "__main__":
    batchsize=1

    real = [[0.0, 1.0], [1.0, 0.0]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    negation = tf.complex(real, imag)

    real = [[1.0 , 0.0], [0.0, -1.0]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    phasechange = tf.complex(real, imag)

    real = [[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    hadamard = tf.complex(real, imag)

    real = tf.random_normal([2,batchsize],0,1)
    imag = tf.random_normal([2,batchsize],0,1)
    #real = [1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)]
    #imag = [0.0, 0.0]


    temp = tf.complex(real, imag)
    vectorrand= tf.div(temp,tf.norm(temp,axis =0))

    real = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    imag = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    cnot = tf.complex(real, imag)

    real = [[1.0, 0.0, 0.0, 0.0]]
    imag = [[0.0, 0.0, 0.0, 0.0]]
    zerozerocte = tf.complex(real, imag)


    #ci=tf.round(inputbatch*10)
    #f0, f1, f2, f3=tf.split(ci,4,axis=1)
    real = [[1 / tf.sqrt(2.0), 0,0, -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0, 0.0, 0.0]]
    phiplus = tf.transpose( tf.complex(real,imag))

    real = [[1 / tf.sqrt(2.0), 0, 0, -1 / tf.sqrt(2.0)],[1 / tf.sqrt(2.0), 0, 0, -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]
    phiplus2 = tf.transpose(tf.complex(real, imag))

    phipluscte=tf.identity(phiplus)
    for x in range(0,batchsize-1):
        phipluscte= tf.concat([phipluscte,phiplus],axis=1)



    #zerozero = unionlayer.inttoqubit(tf.zeros([batchsize]) ,2)

    #phiplusbatch=tf.multiply(tf.transpose(phiplus),tf.ones([1,1024],dtype="complex64"))

    init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**2, 2**2], initializer=init)
    real= tf.eye(2**2, dtype="float32")
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)


    #real = tf.get_variable("w", [2 ** 1, 2 ** 1], initializer=init)
    real= tf.eye(2**1, dtype="float32")
    imag = tf.zeros_like(real, dtype="float32")
    param1 = tf.complex(real, imag)

    learningrate=0.00
    momentum=0.0

    gatecnot = genericQGate.genericQGate(cnot, 2, 2, 0, 0, 0)

    gatecnot1 = genericQGate.genericQGate(cnot, 3, 2, 1, 0, 0)

    gate0 = genericQGate.genericQGate(cnot, 3, 2, 1, learningrate,momentum)

    gateh = genericQGate.genericQGate(hadamard, 3, 1, 0, learningrate, momentum)

    gate1 = genericQGate.genericQGate(negation, 1, 1, 0, learningrate, momentum)

    gate2 = genericQGate.genericQGate(phasechange, 1, 1, 0, learningrate, momentum)

    prep=unionlayer.join(phipluscte,vectorrand)

    o0=gateh.forward(gate0.forward(prep))

    o1,r0=lazymeasure.lazymeasure(o0)

    o2, r1 = lazymeasure.lazymeasure(o1)


    #o3=gate1.forward(unionlayer.join(unionlayer.inttoqubit(r0,1),o2))

    #o4,r2 = lazymeasure.lazymeasure(o3)
    o3 = tf.transpose(tf.where(r1 > 0.9,  tf.transpose(gate1.forward(o2)),   tf.transpose(o2)))

    o4 = tf.transpose(tf.where(r0 > 0.9,   tf.transpose(gate2.forward(o3)),   tf.transpose(o3)))

    #o5 = gate2.forward(unionlayer.join(unionlayer.inttoqubit(r1, 1), o4))

    #o6, r3 = lazymeasure.lazymeasure(o3)

    out=o4

    #out = o0
    measure=[r0,r1]


    target = vectorrand
    cost=lossfunction.fidelity(out,target)
    costm = lossfunction.msq(out, target)
    updates=[]
    #for gates in gate2.gatelist:
    updates.append(gate0.sgd(cost))
    #updates.append(gate1.sgd(cost))
    #updates.append(gate2.sgd(cost))
    #updates = tf.gradients(ys=cost, xs=gate0.param)

    scost = tf.summary.scalar(name='cost', tensor=cost)
    scostm = tf.summary.scalar(name='costm', tensor=costm)

    flag = 1
    max = 0
    itermax = 0
    summaries = tf.summary.merge_all()
    f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        summary_writer =tf.summary.FileWriter("board\\teleport\\log_" + time.strftime("%Y%m%d-%H%M%S"))
        start = tf.global_variables_initializer()
        sess.run(start)
        for i in range(10):
            #print("iter "+str(i))
            fd,cm,up,s,t,o,rawo,m=sess.run([cost,costm, updates,summaries,target,out,o2,measure])
            #print(fd)
            #print(cm)
            print("iter " + str(i)+"\nfd = "+str(fd)+"\nmsq = "+str(cm))
            print("target\n" + str(t))
            print("output\n" + str(o))
            print("rawo\n" + str(rawo))
            print("measures\n" + str(m))
            summary_writer.add_summary(s, i)
            if (fd > max):
                max = fd
                itermax = i
                f.write("iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm)+ "\nparam: " + str(up) + "\n" + "iter end\n")
        print("max")
        print(max)
        print(itermax)
        f.write(
            "iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm) + "\nparam: " + str(up) + "\n" + "iter end\n")


    #cir = subcircuit.ArityFillCircuit(10, 2, 10, "test")
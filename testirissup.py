import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction
import iris_data
import time

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels



if __name__ == "__main__":
    train, test = iris_data.load_data()
    features, labels = train
    dataset = iris_data.train_input_fn(features, labels,150,None)
    dataset= dataset.map(pack_features_vector)
    inputbatch,targetbatch = dataset.make_one_shot_iterator().get_next()
    ci=tf.round(inputbatch*10)
    f0, f1, f2, f3=tf.split(ci,4,axis=1)
    q0 = unionlayer.inttoqubit(f0, 8)
    q1 = unionlayer.inttoqubit(f1, 8)
    q2 = unionlayer.inttoqubit(f2, 8)
    q3 = unionlayer.inttoqubit(f3, 8)





    #real = tf.get_variable("v", [2**14, 2**14], initializer=init)
    init = tf.orthogonal_initializer
    #real = tf.eye(2**8, dtype="float32")
    real = tf.get_variable("v", [2 ** 8, 2 ** 8], initializer=init)
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    #real = tf.eye(2**2, dtype="float32")
    real = tf.get_variable("t", [2 ** 1, 2 ** 1], initializer=init)
    imag = tf.zeros_like(real, dtype="float32")
    param2 = tf.complex(real, imag)

    learningrate = 0.001
    momentum = 0.01

    cir = subcircuit.ArityFillCircuit(8, 2, 8, "test0", learningrate, momentum)
    cir1 = subcircuit.ArityFillCircuit(7, 2, 2, "test1", learningrate, momentum)
    cir2 = subcircuit.ArityFillCircuit(6, 2, 2, "test2", learningrate, momentum)
    cir3 = subcircuit.ArityFillCircuit(5, 2, 2, "test3", learningrate, momentum)
    cir4 = subcircuit.ArityFillCircuit(4, 2, 2, "test4", learningrate, momentum)
    cir5 = subcircuit.ArityFillCircuit(3, 2, 2, "test5", learningrate, momentum)
    #g1 = genericQGate.genericQGate(param2, 8, 8, 0, learningrate, momentum)
    #g1 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)
    #g2 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)
    #g3 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)

    #g0 = genericQGate.genericQGate(param2, 8, 1, 0, learningrate, momentum)


    o0 = cir.forward(q3)
    i1,r1 = lazymeasure.lazymeasure(o0)
    o1 = cir1.forward(i1)
    i2, r2 = lazymeasure.lazymeasure(o1)
    o2 = cir2.forward(i2)
    i3, r3 = lazymeasure.lazymeasure(o2)
    o3 = cir3.forward(i3)
    i4, r4 = lazymeasure.lazymeasure(o3)
    o4 = cir4.forward(i4)
    i5, r5 = lazymeasure.lazymeasure(o4)
    o5 = cir5.forward(i5)
    i6, r6 = lazymeasure.lazymeasure(o5)



    t = tf.pow(2.0, tf.cast(targetbatch, dtype="float32"))

    #target = unionlayer.inttoqubit(tf.pow(2.0, tf.cast(targetbatch, dtype="float32")), 3)
    target2 = unionlayer.inttoqubit( tf.cast(targetbatch, dtype="float32"),2)

    cost=lossfunction.fidelity(i6,target2)
    updates=[cir.update(cost)]
    costm = lossfunction.msq(i6, target2)

    scost = tf.summary.scalar(name='cost', tensor=cost)
    scostm = tf.summary.scalar(name='costm', tensor=costm)

    summaries = tf.summary.merge_all()

    f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    max = 0
    itermax = 0
    with tf.Session() as sess:
        start = tf.global_variables_initializer()
        summary_writer = tf.summary.FileWriter("board\\iris\\log_" + time.strftime("%Y%m%d-%H%M%S"))
        sess.run(start)
        for i in range(1000000):
            print("iter "+str(i))
            sess.run([inputbatch, targetbatch])
            fd,cm,up,s=sess.run([cost,costm,updates,summaries])
            summary_writer.add_summary(s, i)
            print(fd)
            #print(up)
            #print(fd)
            print("iter end")
            if (fd > max):
                max = fd
                itermax = i
                f.write("iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm) + "\nparam: " + str(
                    up) + "\n" + "iter end\n")

        f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))








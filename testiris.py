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

    q0 = unionlayer.inttoqubit(f0, 7)
    q1 = unionlayer.inttoqubit(f1, 7)
    q2 = unionlayer.inttoqubit(f2, 7)
    q3 = unionlayer.inttoqubit(f3, 7)

    init = tf.orthogonal_initializer
    real = tf.get_variable("v", [2**7, 2**7], initializer=init)
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    learningrate=0.001
    momentum=0.01

    g0 = genericQGate.genericQGate(param, 7, 7, 0, learningrate, momentum)

    g1 = genericQGate.genericQGate(param, 7, 7, 0, learningrate, momentum)

    g2 = genericQGate.genericQGate(param, 7, 7, 0, learningrate, momentum)

    g3 = genericQGate.genericQGate(param, 7, 7, 0, learningrate, momentum)

    o0 = g0.forward(q0)
    o1 = g1.forward(q1)
    o2 = g2.forward(q2)
    o3 = g3.forward(q3)

    for x in range(0, 3):
        o0,r0 = lazymeasure.lazymeasure(o0)
        o1,r1 = lazymeasure.lazymeasure(o1)
        o2,r2 = lazymeasure.lazymeasure(o2)
        o3,r3 = lazymeasure.lazymeasure(o3)

    i01 = unionlayer.join(o0,o1)
    i23 = unionlayer.join(o2,o3)

    real = tf.get_variable("o", [2 ** 8, 2 ** 8], initializer=init)
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    g01 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)

    g23 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)

    o01 = g01.forward(i01)
    o23 = g23.forward(i23)

    for x in range(0, 4):
        o01, r01 = lazymeasure.lazymeasure(o01)
        o23, r23 = lazymeasure.lazymeasure(o23)

    iend = unionlayer.join(o01, o23)

    gend = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)

    out = gend.forward(iend)

    #t=tf.pow(2.0,tf.cast(targetbatch,dtype="float32"))

    #target = unionlayer.inttoqubit(tf.pow(2.0,tf.cast(targetbatch,dtype="float32")),3)
    target2 = unionlayer.inttoqubit(tf.cast(targetbatch, dtype="float32"), 2)


    for x in range(0, 6):
        out, rm = lazymeasure.lazymeasure(out)

    cost=lossfunction.fidelity(out,target2)
    costm = lossfunction.msq(out, target2)
    updates=[]
    for gates in gend.gatelist:
        updates.append(gates.rms(cost))

    scost = tf.summary.scalar(name='cost', tensor=cost)
    scostm = tf.summary.scalar(name='costm', tensor=costm)

    flag = 1
    max = 0
    itermax = 0
    summaries = tf.summary.merge_all()
    f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        summary_writer =tf.summary.FileWriter("board\\iris\\log_" + time.strftime("%Y%m%d-%H%M%S"))
        start = tf.global_variables_initializer()
        sess.run(start)
        for i in range(1):
            print("iter "+str(i))
            fd,cm,up,s=sess.run([cost,costm, updates,summaries])
            print(fd)
            print(cm)
            summary_writer.add_summary(s, i)
            if (fd > max):
                max = fd
                itermax = i
                f.write("iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm)+ "\nparam: " + str(up) + "\n" + "iter end\n")

    #cir = subcircuit.ArityFillCircuit(10, 2, 10, "test")
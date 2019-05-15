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
    #ci=tf.round(inputbatch*10)
    #f0, f1, f2, f3=tf.split(ci,4,axis=1)
    normedinputreal= tf.cast(tf.div(inputbatch,tf.norm(inputbatch,axis =0)),dtype="float32")
    imag = tf.zeros_like(normedinputreal, dtype="float32")
    normedinput = tf.transpose(tf.complex(normedinputreal, imag))


    init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**8, 2**8], initializer=init)
    real= tf.eye(2**8, dtype="float32")
    imag = tf.zeros_like(real, dtype="float32")
    param = tf.complex(real, imag)

    learningrate=0.01
    momentum=0.01

    gate0 = genericQGate.genericQGate(param, 8, 8, 0, learningrate,momentum)

    gate1 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)

    gate2 = genericQGate.genericQGate(param, 8, 8, 0, learningrate, momentum)


    copy0 = unionlayer.join(normedinput,normedinput)
    copy1 = unionlayer.join(copy0,normedinput)
    full = unionlayer.join(copy1,unionlayer.inttoqubit(tf.zeros_like(targetbatch, dtype="float32"),2))

    o0=gate0.forward(full)


    for x in range(0,6):
       o0,r = lazymeasure.lazymeasure(o0)

    full1=unionlayer.join(copy1,o0)
    o1=gate1.forward(full1)

    for x in range(0, 6):
        o1, r = lazymeasure.lazymeasure(o1)

    full2 = unionlayer.join(copy1, o1)
    o2 = gate2.forward(full2)

    for x in range(0, 6):
        o2, r = lazymeasure.lazymeasure(o2)


    out=o2

    target = unionlayer.inttoqubit(tf.cast(targetbatch, dtype="float32"), 2)
    cost=lossfunction.fidelity(out,target)
    costm = lossfunction.msq(out, target)
    updates=[]
    for gates in gate2.gatelist:
        updates.append(gates.sgd(cost))

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
        for i in range(300000):
            #print("iter "+str(i))
            fd,cm,up,s=sess.run([cost,costm, updates,summaries])
            #print(fd)
            #print(cm)
            print("iter " + str(i)+"\nfd = "+str(fd)+"\nmsq = "+str(cm))
            summary_writer.add_summary(s, i)
            if (fd > max):
                max = fd
                itermax = i
                f.write("iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm)+ "\nparam: " + str(up) + "\n" + "iter end\n")
        print("max")
        print(max)
        print(itermax)



    #cir = subcircuit.ArityFillCircuit(10, 2, 10, "test")
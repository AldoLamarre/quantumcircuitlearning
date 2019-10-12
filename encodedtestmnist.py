import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction
import iris_data
import time
import vectorencoder
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()



def preprocess(input,nbqubits):
    #ctes=vectorencoder.gennormpluscte(castinput,3,cte=0.01)
    ctes = vectorencoder.gencte(input, nbqubits, cte=0.0)
    #print(ctes)
    encodeddata=vectorencoder.encode_vectors(input,nbqubits,ctes)
    colvec =tf.transpose(encodeddata)
    #print(colvec.shape)
    #c = unionlayer.join(colvec, colvec)
    #v = unionlayer.join(c, c)
    #w = unionlayer.join(v, v)
    return colvec

if __name__ == "__main__":
    #tf.enable_eager_execution()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    learningrate = -0.01
    momentum = 0.9
    datasize = 55000
    batch_size = 8
    valid_batch_size = 8
    iterepoch= datasize / batch_size
    itervalid = 5000 / valid_batch_size
    nbqubits= 18
    targetnbqubit=4
    aritycircuitsize= 9
    aritycircuitdepth= 13

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int64, shape=[None])
    labels = y
    # 01234 v 56789
    #labels = tf.clip_by_value(y,4,5)
    #labels -= 4


    iter = tf.cast(tf.placeholder(tf.int32, shape=()),tf.float32)

    # lrdecay =learningrate
    lrdecay= tf.complex(learningrate / tf.add(iter/100 , 1.0),0.0)
    #lrdecay = learningrate*tf.complex(tf.pow(0.5,tf.floor(iter / 2)), 0.0)

    #init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**nbqubits, 2**nbqubits], initializer=init)
    #real = tf.eye(2 ** nbqubits, dtype="float32")
    #imag = tf.zeros_like(real, dtype="float32")
    #param = tf.complex(real, imag)

    vectorinputs = preprocess(x,nbqubits)

    #cir = subcircuit.ArityFillCircuit(nbqubits, 8, 6, "test0", learningrate, momentum)
    cir = subcircuit.QuincunxCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0")
    #cir = subcircuit.ArityHalfCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    out = cir.forward(vectorinputs)

    outvalid = cir.forward_nesterov_test(vectorinputs, 1, True)
    #outvalid = cir.forward_nesterov_test(vectorinputs)

    #y=tf.transpose(tf.cast(y, dtype="float32"))
    labels = tf.transpose(tf.cast(labels, dtype="float32"))

    #target = unionlayer.onehotqubits(tf.cast(labels_batch, dtype="float32"), targetnbqubit)
    target = unionlayer.inttoqubit(labels, targetnbqubit)
    #print(out.shape)
    #print(target.shape)
    fd_list = lossfunction.fidelity_partial_list(out, target,nbqubits,targetnbqubit)
    #cost=tf.reduce_mean(fd_list)
    majmetric = lossfunction.majority_metric(fd_list)
    maxmetric = lossfunction.max_metric(out, labels,nbqubits,targetnbqubit,10,"inttoqubit")
    cost=lossfunction.cross_entropy(out, target,nbqubits,targetnbqubit)

    fd_list_valid = lossfunction.fidelity_partial_list(outvalid, target, nbqubits, targetnbqubit)
    # cost=tf.reduce_mean(fd_list)
    majmetric_valid = lossfunction.majority_metric(fd_list_valid)
    maxmetric_valid = lossfunction.max_metric(outvalid, labels, nbqubits, targetnbqubit, 10, "inttoqubit")
    cost_valid = lossfunction.cross_entropy(outvalid, target, nbqubits, targetnbqubit)
    mcls_valid = lossfunction.maxclass(outvalid, labels, nbqubits, targetnbqubit, 16, "inttoqubit")

    tmaxmetric_valid = lossfunction.max_metric(outvalid, labels, nbqubits, targetnbqubit, 16, "inttoqubit")

    updates = []
    #for gates in cir.gatelist:
       #updates.append(gates.sgd(cost,lrdecay))
    for i in range(1, len(cir.gatelist)):
        updates.append(cir.gatelist[len(cir.gatelist) - i].sgdnesterov(cost, lrdecay,momentum))

    scost = tf.summary.scalar(name='cost', tensor=cost)






    flag = 1
    max = 0
    min = 10
    itermax = 0
    iterbest =0
    fdloop = 0
    mxmloop=0
    mjmloop = 0
    epoch = 0
    summaries = tf.summary.merge_all()
    f = open("logmnist\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("board\\mnist\\log_" + time.strftime("%Y%m%d-%H%M%S"))

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()


        start = tf.global_variables_initializer()
        sess.run(start)
        for i in range(600000):
            # print("iter "+str(i))

            input_batch, labels_batch = mnist.train.next_batch(batch_size)
            input_batch = input_batch / 255

            #print(labels_batch)
            fd,mjm,mxm, up, s = sess.run([cost,majmetric,maxmetric, updates, summaries],feed_dict={x: input_batch,y: labels_batch,iter: epoch}, options=options, run_metadata=run_metadata)
            fdloop += fd
            mjmloop += mjm
            mxmloop += mxm
            # print(fd)
            # print(cm)
            if  i == 0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                g = open("timeline\\mnist\\log_timeline_" +str(batch_size)+ "_" + str(nbqubits) + "_" +str(targetnbqubit) + "_"
                         + str(aritycircuitsize) + "_" + str(aritycircuitdepth) +"_"
                         + time.strftime("%Y%m%d-%H%M%S") + ".json", 'x')
                g.write(chrome_trace)
                g.close()
            if (i + 1)  % 50 == 0 and epoch <4:
                dem=(i - (epoch * iterepoch) + 1)
                print("iter " + str(i) + "\n-log(fd) = " + str(fdloop/dem))
                print( "majority accuracy = " + str(mjmloop /dem))
                print(" max accuracy = " + str(mxmloop / dem))
                #print(labels_batch)

                #f.write("iter loss list: \n" + str(fd_list)+"\n")
                #f.flush()

            if (i + 1) % iterepoch == 0 and i > 0:
                fdloop /= iterepoch
                mjmloop /= iterepoch
                mxmloop /= iterepoch
                epoch += 1
                print("epoch " + str(epoch) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                + "\nmax accuracy = " + str(mxmloop) )
                summary_writer.add_summary(s, epoch)
                if fdloop < min:
                    min = fdloop
                    iterbest = epoch
                    f.write("iter: " + str(epoch) + "\nfd: " + str(fdloop) + "\nparam: " + str(
                        up) + "\n" + "iter end\n")
                    f.flush()
                fdloop = 0.0
                mxmloop = 0.0
                mjmloop = 0.0
                tmmloop = 0.0

                # input_batch, labels_batch = mnist.validation.images ,mnist.validation.labels
                for i in range(0, 5000 // valid_batch_size):
                    input_batch, labels_batch = mnist.validation.next_batch(valid_batch_size)
                    input_batch = input_batch / 255
                    fd_valid, mjm_valid, mxm_valid, tmm_valid, fdlist_valid, = sess.run(
                        [cost_valid, majmetric_valid, maxmetric_valid, tmaxmetric_valid, fd_list_valid],
                        feed_dict={x: input_batch, y: labels_batch, iter: epoch}, options=options,
                        run_metadata=run_metadata)

                    fdloop += fd_valid
                    mjmloop += mjm_valid
                    mxmloop += mxm_valid
                    tmmloop += tmm_valid

                fdloop /= itervalid
                mjmloop /= itervalid
                mxmloop /= itervalid
                tmmloop /= itervalid

                print(
                    "Valid epoch " + str(epoch) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))

                f.write(
                    "Valid epoch: " + str(epoch) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(
                        tmmloop) + "\n" + "test epoch end\n")

                fdloop = 0.0
                mxmloop = 0.0
                mjmloop = 0.0
                tmmloop = 0.0
                #fdloop = 0
                #mxmloop = 0
                #mjmloop = 0
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as g:
            g.write(chrome_trace)

        #print("max")
        #print(max)
        #print(itermax)
        print("min")
        print(min)
        print(iterbest)
        f.close()

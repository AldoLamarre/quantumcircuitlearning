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
    learningrate = -0.1
    momentum = 0.9
    datasize = 60000
    batch_size = 24
    iterepoch= datasize / batch_size
    nbqubits=10
    targetnbqubit=4
    aritycircuitsize=7
    aritycircuitdepth= 7

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int64, shape=[None])
    labels = y
    # 01234 v 56789
    #labels = tf.clip_by_value(y,4,5)
    #labels -= 4


    iter = tf.cast(tf.placeholder(tf.int32, shape=()),tf.float32)

    # lrdecay =learningrate
    lrdecay= tf.complex(learningrate / tf.add(iter , 1.0),0.0)
    #lrdecay = learningrate*tf.complex(tf.pow(0.5,tf.floor(iter / 2)), 0.0)

    #init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**nbqubits, 2**nbqubits], initializer=init)
    #real = tf.eye(2 ** nbqubits, dtype="float32")
    #imag = tf.zeros_like(real, dtype="float32")
    #param = tf.complex(real, imag)

    vectorinputs = preprocess(x,10)

    #cir = subcircuit.ArityFillCircuit(nbqubits, 8, 6, "test0", learningrate, momentum)
    cir = subcircuit.ArityFillCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    out = cir.forward(vectorinputs)
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

    updates = []
    for gates in cir.gatelist:
       updates.append(gates.sgd(cost,lrdecay))

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
        for i in range(150000):
            # print("iter "+str(i))
            input_batch, labels_batch = mnist.train.next_batch(batch_size)
            input_batch = input_batch / 255


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
                print("iter " + str(i) + "\nfd = " + str(fdloop/dem))
                print( "majority accuracy = " + str(mjmloop /dem))
                print(" max accuracy = " + str(mxmloop / dem))
                #print(labels_batch)

                f.write("iter loss list: \n" + str(fd_list)+"\n")
                f.flush()

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
                    # fdtest = sess.run([costtest])
                    # print("iter test " + str(epoch) + "\nfd  test= " + str(fdtest))
                    # f.write("iter test: " + str(epoch) + "\nfd test: " + str(fdtest) + "\nparam: " + str(
                    #     up) + "\n" + "iter test end\n")
                # if epoch % 10 == 0 or epoch >= 130:
                #     fdtest, ttest, itest, otest, fdlist = sess.run(
                #         [costtest, targettest, testready, outtest, fd_test_list])
                #     print("iter test " + str(epoch) + "\nfd  test= " + str(fdtest))
                #     f.write("iter test:" + str(epoch) + "\nfd test: " + str(fdtest) + "\nparam: " + str(
                #         up) + "\n" + "iter test end\n")
                #
                #     print("input test " + str(itest) + "\ntarget" + str(ttest) + "\nout  test= " + str(otest) +
                #           "\nfd_list  test= " + str(fdlist) + "\ntest end\n")
                #     f.write("input test " + str(itest) + "\ntarget" + str(ttest) + "\nout  test= " + str(otest) +
                #             "\nfd_list  test= " + str(fdlist) + "\ntest end\n")
                fdloop = 0
                mxmloop = 0
                mjmloop = 0
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

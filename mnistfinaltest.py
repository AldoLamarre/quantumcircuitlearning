#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#from pandas.conftest import axis

import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction
import iris_data
import time
import numpy as np
import vectorencoder
import math

from tensorflow.python.client import timeline
import tensorflow_datasets as tfds

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def state_activation_0(x):
    #i*x
    #x = tf.complex(x,tf.zeros_like(x))
    cx= tf.complex(0.0,1.0)*tf.complex(x,tf.zeros_like(x))
    #cx= tf.complex(0.0,1.0)*x
    vect = tf.math.exp(cx)
    tmp = tf.multiply(tf.conj(vect),vect)
    dem = tf.rsqrt(tf.reduce_sum(tmp,axis=1))
    #dem = 1/(tf.reduce_sum(tmp ** 2, axis=1))
    #dem= 1/(tf.norm(vect,axis=1))
    tmp= tf.einsum('ij,i->ij',vect,dem)
    #tmp = dem*vect
    return tmp


def preprocess(input, nbqubits):
    ctes = vectorencoder.gencte(input, nbqubits, cte=0.0)
    encodeddata = vectorencoder.encode_vectors(input, nbqubits, ctes)
    colvec = tf.transpose(encodeddata)
    return colvec


class nnet():
    def __init__(self, size, qubits,linear):
        self.linear=linear
        init = tf.initializers.he_normal()
        init2 = tf.initializers.glorot_normal()
        init3 = tf.initializers.zeros()
        # size = 1024

        self.w0 = tf.get_variable("nnw0", [784, size], initializer=init)
        self.w1 = tf.get_variable("nnw1", [size, size], initializer=init)
        self.b0 = tf.get_variable("nnb0", [size], initializer=init3)
        self.b1 = tf.get_variable("nnb1", [size], initializer=init3)

        self.bfr = tf.get_variable("nnbfr", [2 ** qubits], initializer=init3)

        self.wmlp = tf.get_variable("nnw0f", [size, 2 ** qubits], initializer=init2)

        self.nnparamlist = []

        self.nnparamlist.append(self.w1)
        self.nnparamlist.append(self.b1)
        self.nnparamlist.append(self.w0)
        self.nnparamlist.append(self.b0)
        self.nnparamlist.append(self.bfr)

        self.nnparamlist.append(self.wmlp)

    def get_nnparamlist(self):
        return self.nnparamlist

    def state_activation_0(self, x):
        # i*x
        # x = tf.complex(x,tf.zeros_like(x))
        cx = tf.complex(0.0, 1.0) * tf.complex(x, tf.zeros_like(x))
        # cx= tf.complex(0.0,1.0)*x
        vect = tf.math.exp(cx)
        tmp = tf.multiply(tf.conj(vect), vect)
        dem = tf.rsqrt(tf.reduce_sum(tmp, axis=1))
        # dem = 1/(tf.reduce_sum(tmp ** 2, axis=1))
        # dem= 1/(tf.norm(vect,axis=1))
        tmp = tf.einsum('ij,i->ij', vect, dem)
        # tmp = dem*vect
        return tmp

    def forward(self, x,):
        if self.linear:
            l0 = (tf.matmul(x, self.w0) + self.b0)
            l1 = (tf.matmul(l0, self.w1) + self.b1)
        else:
            l0 = tf.nn.relu(tf.matmul(x, self.w0) + self.b0)
            l1 = tf.nn.relu(tf.matmul(l0, self.w1) + self.b1)

        ltr = (tf.matmul(l1, self.wmlp) + self.bfr)
        lf = self.state_activation_0(ltr)

        return lf

def lossandmetric(out,labels,nbqubits,targetnbqubit):
    labels = tf.transpose(tf.cast(labels, dtype="float32"))
    target = unionlayer.inttoqubit(labels, targetnbqubit)

    fd_list = lossfunction.fidelity_partial_list(out, target, nbqubits, targetnbqubit)
    # cost=tf.reduce_mean(fd_list)
    majmetric = lossfunction.majority_metric(fd_list)
    # 10 is the class number
    maxmetric = lossfunction.max_metric(out, labels,  nbqubits, targetnbqubit, 10, "inttoqubit")
    cost = lossfunction.cross_entropy(out, target, nbqubits, targetnbqubit)
    mcls = lossfunction.maxclass(out, labels,  nbqubits, targetnbqubit, 2 ** targetnbqubit, "inttoqubit")

    tmaxmetric = lossfunction.max_metric(out, labels, nbqubits, targetnbqubit,  2 ** targetnbqubit, "inttoqubit")

    return fd_list,cost,majmetric,maxmetric,tmaxmetric,mcls,

def runmodel(hybrid,linear,maxepoch):
    # tf.enable_eager_execution()
    tf.reset_default_graph()
    stringtype = "a"
    if hybrid:
        if linear:
            stringtype = "hybridlinear"
        else:
            stringtype = "hybridnnet"
    else:
        stringtype = "purequantum"

    learningrate = -0.01
    momentum = 0.9
    datasize = 55000
    datasize_valid = 5000
    datasize_test = 5000
    nnsize=1024
    batch_size = 8
    valid_batch_size = 64
    test_batch_size = 64
    iterepoch = math.ceil(datasize / batch_size)
    itervalid = math.ceil(datasize_valid / valid_batch_size)
    itertest = math.ceil(datasize_test / test_batch_size)
    nbqubits = 18
    targetnbqubit = 4
    aritycircuitsize = 9
    aritycircuitdepth = 13

    trainmnsit, validmnsit = tfds.load("mnist:3.*.*", data_dir="dataset\\",
                                       split=['train[0:55000]', 'train[55000:60000]'])

    testmnsit, = tfds.load("mnist:3.*.*", data_dir="dataset\\", split=['test[0:5000]'])

    # trainemnistletters, = tfds.load("emnist/letters:3.*.*", data_dir="dataset\\", split=['train[0:'+str(datasize_emnist_letter)+']'] )

    # trainkmnist, = tfds.load("kmnist:3.*.*", data_dir="dataset\\", split=['train[0:'+str(datasize_kmnist)+']'])

    batch = trainmnsit.batch(batch_size).prefetch(3).make_initializable_iterator()

    batch_valid = validmnsit.batch(valid_batch_size).prefetch(5).make_initializable_iterator()

    batch_test = testmnsit.batch(test_batch_size).prefetch(5).make_initializable_iterator()

    current = batch.get_next()
    current_valid = batch_valid.get_next()
    current_test = batch_test.get_next()

    x, y = tf.cast(tf.reshape(current["image"], [tf.shape(current["image"])[0], 784]), dtype="float32") / 256, current[
        "label"]

    x_valid, y_valid = tf.cast(tf.reshape(current_valid["image"], [tf.shape(current_valid["image"])[0], 784]),
                               dtype="float32") / 256, current_valid["label"]
    x_test, y_test = tf.cast(tf.reshape(current_test["image"], [tf.shape(current_test["image"])[0], 784]),
                             dtype="float32") / 256, current_test["label"]


    labels = y
    iter = tf.cast(tf.placeholder(tf.int32, shape=()), tf.float32)
    # iter = 1
    # lrdecay =learningrate


    #lrdecay = tf.complex(learningrate / tf.add(iter / 10, 1.0), 0.0)
    # / 100
    lrdecay = tf.complex(learningrate / tf.add(iter / 100, 1.0), 0.0)
    #first_decay_steps = 1000
    #lrdecay=tf.complex(tf.train.cosine_decay_restarts(learningrate,iter,first_decay_steps),0.0)

    # lrdecay = learningrate*tf.complex(tf.pow(0.5,tf.floor(iter / 2)), 0.0)
    nnparamlist = tf.complex(0.0, 1.0)
    if hybrid:
        nn = nnet(nnsize,nbqubits,linear)
        nnparamlist = nn.get_nnparamlist()

        lf = nn.forward(x)

        nninput = tf.transpose(lf)
        nninput_valid = tf.transpose(nn.forward(x_valid))
        nninput_test = tf.transpose(nn.forward(x_test))
    else:
        nninput = preprocess(x, nbqubits)
        nninput_valid = preprocess(x_valid, nbqubits)
        nninput_test = preprocess(x_test, nbqubits)


    # nninput_emnistletters = tf.transpose(nn.forward(x_emnistletters))
    # nninput_kmnist = tf.transpose(nn.forward(x_kmnist))

    # print(nninput)
    # print(lf)

    # cir = subcircuit.ArityFillCircuit(nbqubits, 8, 6, "test0", learningrate, momentum)
    # cir = subcircuit.ArityFillCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    # cir = subcircuit.ArityHalfCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    cir = subcircuit.QuincunxCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0")

    # out = cir.forward_two_inputs(nninput,stateinput)
    # out = cir.forward_two_inputs(nninput, nninput)
    # out = cir.forward(nninput)
    # initopt =  cir.decompose()

    out = cir.forward(nninput, 1)

    # print(tf.norm(out, axis=0))
    # print(np.linalg.norm(out, ord=2, axis=0))

    # for x in range(0,8):

    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)
    # out = cir.recursion(out)

    outvalid = cir.forward_nesterov_test(nninput_valid, 1, True)
    outtest = cir.forward_nesterov_test(nninput_test, 1, False)
    # out_emnistletters = cir.forward_nesterov_test(nninput_emnistletters, 1, False)
    # out_kmnist = cir.forward_nesterov_test(nninput_kmnist, 1, False)
    # print(tf.norm(out, axis=0))
    # out = tf.div(out, )
    # print(out.shape)
    # y=tf.transpose(tf.cast(y, dtype="float32"))
    labels = tf.transpose(tf.cast(labels, dtype="float32"))
    labels_valid = tf.transpose(tf.cast(y_valid, dtype="float32"))
    labels_test = tf.transpose(tf.cast(y_test, dtype="float32"))

    # target = unionlayer.onehotqubits(tf.cast(labels_batch, dtype="float32"), targetnbqubit)
    target = unionlayer.inttoqubit(labels, targetnbqubit)
    target_valid = unionlayer.inttoqubit(labels_valid, targetnbqubit)
    #target_test = unionlayer.inttoqubit(labels_test, targetnbqubit)
    # print(out.shape)
    # print(target.shape)
    # print(out)
    # print("target")
    # print(target)
    fd_list = lossfunction.fidelity_partial_list(out, target, nbqubits, targetnbqubit)
    # print("fd1")
    # print(fd_list)
    # target2 = -1.0*(target -tf.ones_like(target) )
    # print("target2")
    # print(target2)
    # fd_list2 = lossfunction.fidelity_partial_list(out, target2, nbqubits, targetnbqubit)
    # print("fd2")
    # print(fd_list2)

    # cost=tf.reduce_mean(fd_list)
    majmetric = lossfunction.majority_metric(fd_list)
    maxmetric = lossfunction.max_metric(out, labels, nbqubits, targetnbqubit, 10, "inttoqubit")
    cost = lossfunction.cross_entropy(out, target, nbqubits, targetnbqubit)
    mcls = lossfunction.maxclass(out, labels, nbqubits, targetnbqubit, 10, "inttoqubit")

    tmaxmetric = lossfunction.max_metric(out, labels, nbqubits, targetnbqubit, 16, "inttoqubit")

    fd_list_valid = lossfunction.fidelity_partial_list(outvalid, target_valid, nbqubits, targetnbqubit)
    # cost=tf.reduce_mean(fd_list)
    majmetric_valid = lossfunction.majority_metric(fd_list_valid)
    maxmetric_valid = lossfunction.max_metric(outvalid, labels_valid, nbqubits, targetnbqubit, 10, "inttoqubit")
    cost_valid = lossfunction.cross_entropy(outvalid, target_valid, nbqubits, targetnbqubit)
    mcls_valid = lossfunction.maxclass(outvalid, labels_valid, nbqubits, targetnbqubit, 16, "inttoqubit")

    tmaxmetric_valid = lossfunction.max_metric(outvalid, labels_valid, nbqubits, targetnbqubit, 16, "inttoqubit")

    fd_list_test, cost_test, majmetric_test, maxmetric_test, tmaxmetric_test, mcls_test \
        = lossandmetric(outtest, labels_test, nbqubits, targetnbqubit)


    updates = []
    # for gates in cir.gatelist:
    # updates.append(gates.sgd(cost,lrdecay))
    for i in range(1, len(cir.gatelist)):
        updates.append(cir.gatelist[len(cir.gatelist) - i].sgdnesterov(cost, lrdecay,momentum))

    # opt=tf.train.GradientDescentOptimizer(lrdecay)
    # grads = tf.gradients(cost,nnparamlist)
    # nnup = opt.apply_gradients(cost, None, nnparamlist)
    # for i in range(0,len(nnparamlist)):
    # nnparamlist[i]-=1*grads[i]
    grads= tf.complex(0.0, 1.0)
    grads1 = tf.complex(0.0, 1.0)
    grads2 = tf.complex(0.0, 1.0)
    if hybrid:
        it = tf.Variable(0, dtype="int64")
        #opt = tf.train.AdamOptimizer()
        #opt = tf.contrib.opt.NadamOptimizer()
        #opt=tf.contrib.optimizer_v2.MomentumOptimizer(-1*tf.real(lrdecay),momentum,use_nesterov=True)

        opt1 = tf.contrib.opt.NadamOptimizer()

        opt2 = tf.contrib.optimizer_v2.MomentumOptimizer(-1 * tf.real(lrdecay), momentum, use_nesterov=True)


        # grads= opt.minimize(cost,it, nnparamlist)
        #grads = opt.minimize(cost, it, nnparamlist)

        grads1 = opt1.minimize(cost, it, nnparamlist)
        grads2 = opt2.minimize(cost, it, nnparamlist)
    # for ng in grads:
    # gsum.append(tf.reduce_sum(ng))

    scost = tf.summary.scalar(name='cost', tensor=cost)
    prev = []
    flag = 1
    max = 0
    min = 10
    itermax = 0
    iterbest = 0
    iterbestvalid = 0
    fdloop = 0.0
    mxmloop = 0.0
    mjmloop = 0.0
    tmmloop = 0.0
    sm = 0
    lup = 0

    # epoch = 0
    summaries = tf.summary.merge_all()
    f = open("logmnist_final\\log_"  +str(stringtype)+time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    h = open("logmnist_final\\param_" + str(stringtype) + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    h.write("\n" + time.strftime("%Y%m%d-%H%M%S"))

    with tf.Session() as sess:
        start = tf.global_variables_initializer()

        sess.run(start)
        summary_writer = tf.summary.FileWriter("board\\notmnist_hybrid\\\log_" + time.strftime("%Y%m%d-%H%M%S"))

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()


        # sess.run(initopt)
        for i in range(maxepoch):
            # print("iter "+str(i))
            sess.run(batch.initializer)

            for j in range(datasize):
                try:

                    # c = sess.run([current])

                    # print(labels_batch)

                    c, fd, mjm, mxm, tmm, mc, up, s, nnup, wnn, fdlist, nin, label_batch, ot = sess.run(
                    [current, cost, majmetric, maxmetric, tmaxmetric, mcls, updates, summaries, grads1, nnparamlist,
                    fd_list, nninput, y, out], feed_dict={iter: i}, options=options, run_metadata=run_metadata)

                    # if j < 50 :
                    #     c, fd, mjm, mxm, tmm, mc, up, s, nnup, wnn, fdlist, nin, label_batch, ot = sess.run(
                    #         [current, cost, majmetric, maxmetric, tmaxmetric, mcls, updates, summaries, grads1, nnparamlist,
                    #         fd_list, nninput, y, out], feed_dict={iter: (i+1)*j}, options=options, run_metadata=run_metadata)
                    #
                    # else:
                    #     c, fd, mjm, mxm, tmm, mc, up, s, nnup, wnn, fdlist, nin, label_batch, ot = sess.run(
                    #         [current, cost, majmetric, maxmetric, tmaxmetric, mcls, updates, summaries, grads2,
                    #          nnparamlist,
                    #          fd_list, nninput, y, out], feed_dict={iter: (i + 1) * j}, options=options,
                    #         run_metadata=run_metadata)


                    fdloop += fd
                    mjmloop += mjm
                    mxmloop += mxm
                    tmmloop += tmm
                    # if i == 0:
                    #     prev.append(wnn[0])
                    #     prev.append(wnn[1])
                    #     prev.append(wnn[2])
                    # else :
                    #     print(np.sum(prev[0]-wnn[0]))
                    #     print(np.sum(prev[1] - wnn[1]))
                    #     print(np.sum(prev[2] - wnn[2]))

                    # print(fd)
                    # print(cm)
                    if i + j == 0:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()



                        g = open(
                            "timeline\\mnist_final\\log_timeline_"+str(stringtype) + str(batch_size) + "_" + str(nbqubits) + "_" + str(
                                targetnbqubit) + "_"
                            + str(aritycircuitsize) + "_" + str(aritycircuitdepth) + "_"
                            + time.strftime("%Y%m%d-%H%M%S") + ".json", 'x')
                        g.write(chrome_trace)
                        g.close()
                    if (j + 1) % 50 == 0 and i < 4:
                        dem = (j + 1)
                        print("iter " + str(j) + "\n-log(fd) = " + str(fdloop / dem))
                        print("majority accuracy = " + str(mjmloop / dem))
                        print(" max accuracy = " + str(mxmloop / dem))
                        print(" true max accuracy = " + str(tmmloop / dem))
                        print(" fdlist")
                        print(fdlist)
                        print(" max class / label")
                        print(mc)
                        print(label_batch)

                        # f.write("iter loss list: \n" + str(fd_list)+"\n")
                        # f.flush()
                    if (j + 1) % 150 == 0 and i < 4:
                        print("n input")
                        print(nin)
                        print(np.linalg.norm(nin, ord=2, axis=0))
                        print(np.linalg.norm(ot, ord=2, axis=0))
                    sm = s
                    lup = up
                except tf.errors.OutOfRangeError:
                    break

            fdloop /= iterepoch
            mjmloop /= iterepoch
            mxmloop /= iterepoch
            tmmloop /= iterepoch
            print(iterepoch)
            # epoch += 1
            print("epoch " + str(i) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                  + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))
            summary_writer.add_summary(sm, i)
            f.write("iter: " + str(i) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop)
                    + "iter end\n")
            h.write("iter: " + str(i) +"\nparam: "
                    + str(lup) + "\n" + "iter end\n")
            f.flush()
            h.flush()

            fdloop = 0.0
            mxmloop = 0.0
            mjmloop = 0.0
            tmmloop = 0.0
            mnistprob_emnistletters = 0.0
            restprob_emnistletters = 0.0
            maxclassrest_emnistletters = 0.0
            mnistprob_kmnist = 0.0
            restprob_kmnist = 0.0
            maxclassrest_kmnist = 0.0
            sess.run(batch_valid.initializer)
            fdlist_valid = 0
            for j in range(datasize_valid):
                try:
                    # problist_emnistletters , probalist_emnistletters
                    fd_valid, mjm_valid, mxm_valid, tmm_valid, fdlist_valid = sess.run(
                        [cost_valid, majmetric_valid, maxmetric_valid, tmaxmetric_valid, fd_list_valid]
                        , feed_dict={iter: i}, options=options, run_metadata=run_metadata)

                    # print(c)
                    fdloop += fd_valid
                    mjmloop += mjm_valid
                    mxmloop += mxm_valid
                    tmmloop += tmm_valid

                except tf.errors.OutOfRangeError:
                    break

            fdloop /= itervalid
            mjmloop /= itervalid
            mxmloop /= itervalid
            tmmloop /= itervalid

            print("Valid epoch " + str(i) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                  + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))
            print("fdlist valid\n" + str(fdlist_valid))
            if fdloop < min:
                min = fdloop
                iterbest = i

            f.write("Valid epoch: " + str(i) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(
                tmmloop) + "\n" + "Valid epoch end\n")


            fdloop = 0.0
            mxmloop = 0.0
            mjmloop = 0.0
            tmmloop = 0.0

            sess.run(batch_test.initializer)
            fdlist_test=0
            for j in range(datasize_test):
                try:
                    fd_test, mjm_test, mxm_test, tmm_test, fdlist_test = sess.run(
                        [cost_test, majmetric_test, maxmetric_test, tmaxmetric_test, fd_list_test]
                        , feed_dict={iter: i}, options=options, run_metadata=run_metadata)

                    fdloop += fd_test
                    mjmloop += mjm_test
                    mxmloop += mxm_test
                    tmmloop += tmm_test

                except tf.errors.OutOfRangeError:
                    break

            fdloop /= itertest
            mjmloop /= itertest
            mxmloop /= itertest
            tmmloop /= itertest

            print("Test epoch " + str(i) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                  + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))
            print("fdlist valid\n" + str(fdlist_test))

            f.write("Test epoch: " + str(i) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(
                tmmloop) + "\n" + "Test epoch end\n")

        print("min")
        print(min)
        print(iterbest)
        f.write("iterbesttrain: " + str(iterbest))
        f.write("iterbestvalid: " + str(iterbest))
        f.close()
        h.close()
        sess.close()

if __name__ == "__main__":
    maxepoch= 60
    runmodel(0, 0, maxepoch)
    #runmodel(1, 1, maxepoch)
    #runmodel(1, 0, maxepoch)
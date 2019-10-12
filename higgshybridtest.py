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
from tensorflow.python.client import timeline
import tensorflow_datasets as tfds


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


def preprocess(current): #for current in  train.batch(batch_size):

    # print(type(current))
    # print(current)
    class_label = current['class_label']
    jet_1_b_tag = current['jet_1_b-tag']
    jet_1_eta = current['jet_1_eta']
    jet_1_phi = current['jet_1_phi']
    jet_1_pt = current['jet_1_pt']
    jet_2_b_tag = current['jet_2_b-tag']
    jet_2_eta = current['jet_2_eta']
    jet_2_phi = current['jet_2_phi']
    jet_2_pt = current['jet_2_pt']
    jet_3_b_tag = current['jet_3_b-tag']
    jet_3_eta = current['jet_3_eta']
    jet_3_phi = current['jet_3_phi']
    jet_3_pt = current['jet_3_pt']
    jet_4_b_tag = current['jet_4_b-tag']
    jet_4_eta = current['jet_4_eta']
    jet_4_phi = current['jet_4_phi']
    jet_4_pt = current['jet_4_pt']
    lepton_eta = current['lepton_eta']
    lepton_pT = current['lepton_pT']
    lepton_phi = current['lepton_phi']
    m_bb = current['m_bb']
    m_jj = current['m_jj']
    m_jjj = current['m_jjj']
    m_jlv = current['m_jlv']
    m_lv = current['m_lv']
    m_wbb = current['m_wbb']
    m_wwbb = current['m_wwbb']
    missing_energy_magnitude = current['missing_energy_magnitude']
    missing_energy_phi = current['missing_energy_phi']
    # jet_1= tf.stack([jet_1_b_tag,jet_1_eta,jet_2_phi,jet_2_pt],axis=0)
    # jet_2 =tf.stack([jet_2_b_tag, jet_2_eta, jet_2_phi, jet_2_pt],axis=0)
    # jet_3 =tf.stack([jet_3_b_tag, jet_3_eta, jet_3_phi, jet_3_pt],axis=0)
    # jet_4 =tf.stack([jet_4_b_tag, jet_4_eta, jet_4_phi, jet_4_pt],axis=0)
    # leprest =tf.stack([lepton_eta, lepton_pT, lepton_phi, missing_energy_magnitude,missing_energy_phi],axis=0)
    # input_batch= tf.stack([jet_1,jet_2,jet_3,jet_4,leprest],axis=0)

    input_batch = tf.transpose(tf.cast(tf.squeeze(
        tf.stack([jet_1_b_tag, jet_1_eta, jet_2_phi, jet_2_pt, jet_2_b_tag, jet_2_eta, jet_2_phi, jet_2_pt,
                               jet_3_b_tag, jet_3_eta, jet_3_phi, jet_3_pt, jet_4_b_tag, jet_4_eta, jet_4_phi, jet_4_pt,
                               lepton_eta, lepton_pT, lepton_phi, missing_energy_magnitude, missing_energy_phi],
                              axis=0)),dtype='float32'))

    return input_batch,class_label


class nnet():
    def __init__(self):
        init = tf.initializers.he_normal()
        init2 = tf.initializers.glorot_normal()
        init3 = tf.initializers.zeros()
        size=512

        self.w0 = tf.get_variable("nnw0", [21, size], initializer=init)
        self.w1 = tf.get_variable("nnw1", [size, size], initializer=init)
        #self.w2 = tf.get_variable("nnw2", [size, size], initializer=init)
        #self.w3 = tf.get_variable("nnw3", [size, size], initializer=init)
        #self.w4 = tf.get_variable("nnw4", [size, size], initializer=init)
        # wf = tf.get_variable("nnwf", [512,16], initializer=init2)
        # wfr = tf.get_variable("nnwfr", [512, 16], initializer=init2)
        # wfc = tf.get_variable("nnwfc", [512, 16], initializer=init2)
        self.b0 = tf.get_variable("nnb0", [size], initializer=init3)
        self.b1 = tf.get_variable("nnb1", [size], initializer=init3)
        #self.b2 = tf.get_variable("nnb2", [size], initializer=init3)
        #self.b3 = tf.get_variable("nnb3", [size], initializer=init3)
        #self.b4 = tf.get_variable("nnb4", [size], initializer=init3)

        # bf = tf.get_variable("nnbf", [16], initializer=init3)
        self.bfr = tf.get_variable("nnbfr", [2 ** 6], initializer=init3)
        # bfc = tf.get_variable("nnbfc", [16], initializer=init3)
        self.wmlp = tf.get_variable("nnw0f", [size, 2 ** 6], initializer=init2)

        self.nnparamlist = []

        #self.nnparamlist.append(self.w4)
        #self.nnparamlist.append(self.b4)

        #self.nnparamlist.append(self.w3)
        #self.nnparamlist.append(self.b3)

        #self.nnparamlist.append(self.w2)
        #self.nnparamlist.append(self.b2)

        self.nnparamlist.append(self.w1)
        self.nnparamlist.append(self.b1)
        self.nnparamlist.append(self.w0)
        self.nnparamlist.append(self.b0)
        # nnparamlist.append(bf)
        # nnparamlist.append(wf)
        # nnparamlist.append(bfc)
        # nnparamlist.append(wfc)
        self.nnparamlist.append(self.bfr)
        # nnparamlist.append(wfr)
        self.nnparamlist.append(self.wmlp)

    def get_nnparamlist(self):
        return self.nnparamlist

    def forward(self,x):
        l0 = tf.nn.relu(tf.matmul(x, self.w0) + self.b0)
        l1 = tf.nn.relu(tf.matmul(l0, self.w1) + self.b1)
        #l2 = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)
        #l3 = tf.nn.relu(tf.matmul(l2, self.w3) + self.b3)
        #l4 = tf.nn.relu(tf.matmul(l3, self.w4) + self.b4)
        # ltr = tf.nn.tanh(tf.matmul(l1, wfr) + bfr)
        # tc = tf.nn.tanh(tf.matmul(l1, wfc) + bfc)
        # lf = state_activation(ltr,ltc)

        #ltr = (tf.matmul(x, self.wmlp) + self.bfr)


        ltr = (tf.matmul(l1, self.wmlp) + self.bfr)

        #ltr = (tf.matmul(l4, self.wmlp) + self.bfr)

        # print(ltr)

        lf = state_activation_0(ltr)

        return lf




if __name__ == "__main__":
    #tf.enable_eager_execution()
    #higgs =
    learningrate = -0.01
    momentum = 0.9
    datasize = 800000
    datasize_valid= 200000
    batch_size = 200
    valid_batch_size=10000
    iterepoch= datasize / batch_size
    itervalid= datasize_valid / valid_batch_size
    nbqubits= 12
    targetnbqubit=1
    aritycircuitsize= 6
    aritycircuitdepth= 13

    data = tfds.load("higgs", data_dir="dataset\\")
    # train, test = data['train'],data['test']
    train = data['train'].take(datasize)
    valid = data['train'].skip(datasize).take(datasize_valid)
    #train = data['train'].take(1024)



    batch = train.batch(batch_size).make_initializable_iterator()

    batch_valid = valid.batch(valid_batch_size).make_initializable_iterator()

    current = batch.get_next()
    current_valid = batch_valid.get_next()

    x,y = preprocess(current)

    #x = tf.placeholder(tf.float32, shape=[None, 21])
    #y = tf.placeholder(tf.int64, shape=[None])

    x_valid, y_valid = preprocess(current_valid)



    #anc = tf.zeros([batch_size,1])
    #anc=tf.zeros_like(y)
    #anc = unionlayer.inttoqubit(anc,targetnbqubit-2)
    # 01234 v 56789
    #labels = tf.clip_by_value(y,4,5)
    #labels -= 4

    #input_batch, labels_batch = mnist.train.next_batch(batch_size)
    #input_batch = input_batch / 255

    #x=input_batch
    #y=labels_batch
    #stateinput = preprocess(x, 10)

    labels = y
    iter = tf.cast(tf.placeholder(tf.int32, shape=()),tf.float32)
    #iter = 1
    #lrdecay =learningrate
    lrdecay= tf.complex(learningrate / tf.add(iter/10 , 1.0),0.0)
    #lrdecay = learningrate*tf.complex(tf.pow(0.5,tf.floor(iter / 2)), 0.0)

    nn=nnet()
    nnparamlist=nn.get_nnparamlist()

    lf=nn.forward(x)


    nninput=tf.transpose(lf)
    nninput_valid=tf.transpose(nn.forward(x_valid))

    #print(nninput)
    #print(lf)


    #cir = subcircuit.ArityFillCircuit(nbqubits, 8, 6, "test0", learningrate, momentum)
    #cir = subcircuit.ArityFillCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    #cir = subcircuit.ArityHalfCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    cir = subcircuit.HalfQuincunxCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0")

    #out = cir.forward_two_inputs(nninput,stateinput)
    #out = cir.forward_two_inputs(nninput, nninput)
    #out = cir.forward(nninput)
    out = cir.forward(nninput,1)

    #print(tf.norm(out, axis=0))
    #print(np.linalg.norm(out, ord=2, axis=0))

    #for x in range(0,8):

    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)

    outvalid = cir.forward_nesterov_test(nninput_valid,1,True)
    #print(tf.norm(out, axis=0))
    #out = tf.div(out, )
    #print(out.shape)
    #y=tf.transpose(tf.cast(y, dtype="float32"))
    labels = tf.transpose(tf.cast(labels, dtype="float32"))
    labels_valid = tf.transpose(tf.cast(y_valid, dtype="float32"))

    #target = unionlayer.onehotqubits(tf.cast(labels_batch, dtype="float32"), targetnbqubit)
    target = unionlayer.inttoqubit(labels, targetnbqubit)
    target_valid = unionlayer.inttoqubit(labels_valid, targetnbqubit)
    #print(out.shape)
    #print(target.shape)
    #print(out)
    #print("target")
    #print(target)
    fd_list = lossfunction.fidelity_partial_list(out, target,nbqubits,targetnbqubit)
    #print("fd1")
    #print(fd_list)
    target2 = -1.0*(target -tf.ones_like(target) )
    #print("target2")
    #print(target2)
    fd_list2 = lossfunction.fidelity_partial_list(out, target2, nbqubits, targetnbqubit)
    #print("fd2")
    #print(fd_list2)


    #cost=tf.reduce_mean(fd_list)
    majmetric = lossfunction.majority_metric(fd_list)
    maxmetric = lossfunction.max_metric(out, labels,nbqubits,targetnbqubit,2,"inttoqubit")
    cost=lossfunction.cross_entropy(out, target,nbqubits,targetnbqubit)
    mcls=lossfunction.maxclass(out, labels,nbqubits,targetnbqubit,2,"inttoqubit")

    tmaxmetric =  lossfunction.max_metric(out, labels,nbqubits,targetnbqubit,2,"inttoqubit")


    fd_list_valid = lossfunction.fidelity_partial_list(outvalid, target_valid,nbqubits,targetnbqubit)
    #cost=tf.reduce_mean(fd_list)
    majmetric_valid = lossfunction.majority_metric(fd_list_valid)
    maxmetric_valid = lossfunction.max_metric(outvalid, labels_valid,nbqubits,targetnbqubit,2,"inttoqubit")
    cost_valid=lossfunction.cross_entropy(outvalid, target_valid,nbqubits,targetnbqubit)
    mcls_valid=lossfunction.maxclass(outvalid, labels_valid,nbqubits,targetnbqubit,2,"inttoqubit")

    tmaxmetric_valid =  lossfunction.max_metric(outvalid, labels_valid,nbqubits,targetnbqubit,2,"inttoqubit")

    updates = []
    #for gates in cir.gatelist:
       #updates.append(gates.sgd(cost,lrdecay))
    for i in range(1, len(cir.gatelist)):
        updates.append(cir.gatelist[len(cir.gatelist) - i].sgdnesterov(cost, lrdecay,momentum))






    #opt=tf.train.GradientDescentOptimizer(lrdecay)
    #grads = tf.gradients(cost,nnparamlist)
    #nnup = opt.apply_gradients(cost, None, nnparamlist)
    #for i in range(0,len(nnparamlist)):
        #nnparamlist[i]-=1*grads[i]

    it=tf.Variable(0,dtype="int64")
    #opt = tf.train.AdamOptimizer()
    opt= tf.contrib.opt.NadamOptimizer()
    #opt=tf.contrib.optimizer_v2.MomentumOptimizer(10*lrdecay,momentum)
    #grads= opt.minimize(cost,it, nnparamlist)
    grads = opt.minimize(cost,it, nnparamlist)
    #for ng in grads:
        #gsum.append(tf.reduce_sum(ng))



    scost = tf.summary.scalar(name='cost', tensor=cost)
    prev = []
    flag = 1
    max = 0
    min = 10
    itermax = 0
    iterbest =0
    fdloop = 0.0
    mxmloop=0.0
    mjmloop = 0.0
    tmmloop = 0.0
    sm=0
    lup=0

    #epoch = 0
    summaries = tf.summary.merge_all()
    f = open("loghiggs_hybrid\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))


    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("board\\higgs_hybrid\\\log_" + time.strftime("%Y%m%d-%H%M%S"))

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start = tf.global_variables_initializer()


        sess.run(start)
        for i in range(200):
            # print("iter "+str(i))
            sess.run(batch.initializer)

            for j in range(datasize):
                try:


                    #c = sess.run([current])

                    #print(labels_batch)
                    c,fd,mjm,mxm,tmm,mc,up, s, nnup, wnn,fdlist, nin,label_batch,ot = sess.run([current,cost,majmetric,maxmetric,tmaxmetric, mcls,updates, summaries,grads,nnparamlist,fd_list,nninput,y,out],feed_dict={iter: i}, options=options, run_metadata=run_metadata)
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
                    if  i+j == 0:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        g = open("timeline\\higgs\\log_timeline_" +str(batch_size)+ "_" + str(nbqubits) + "_" +str(targetnbqubit) + "_"
                                 + str(aritycircuitsize) + "_" + str(aritycircuitdepth) +"_"
                                 + time.strftime("%Y%m%d-%H%M%S") + ".json", 'x')
                        g.write(chrome_trace)
                        g.close()
                    if (j + 1)  % 50 == 0 and i <4:
                        dem=(j + 1)
                        print("iter " + str(j) + "\n-log(fd) = " + str(fdloop/dem))
                        print( "majority accuracy = " + str(mjmloop /dem))
                        print(" max accuracy = " + str(mxmloop / dem))
                        print(" true max accuracy = " + str(tmmloop / dem))
                        print(" fdlist")
                        print(fdlist)
                        print(" max class / label")
                        print(mc)
                        print(label_batch)

                        #f.write("iter loss list: \n" + str(fd_list)+"\n")
                        #f.flush()
                    if (j + 1) % 150 == 0 and i < 4:
                        print("n input")
                        print(nin)
                        print(np.linalg.norm(nin, ord=2,axis=0))
                        print(np.linalg.norm(ot, ord=2, axis=0))
                    sm=s
                    lup=up
                except tf.errors.OutOfRangeError:
                    break


            fdloop /= iterepoch
            mjmloop /= iterepoch
            mxmloop /= iterepoch
            tmmloop /= iterepoch
            #epoch += 1
            print("epoch " + str(i) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
            + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))
            summary_writer.add_summary(sm, i)
            if fdloop < min:
                min = fdloop
                iterbest = i
                f.write("iter: " + str(i) + "\nfd: " + str(fdloop) +  "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop) + "\nparam: "
                        + str(lup) + "\n" + "iter end\n")
                f.flush()

            fdloop = 0.0
            mxmloop = 0.0
            mjmloop = 0.0
            tmmloop = 0.0

            sess.run(batch_valid.initializer)
            for j in range(datasize_valid):
                try:
                    fd_valid, mjm_valid, mxm_valid, tmm_valid, fdlist_valid = sess.run(
                        [cost_valid, majmetric_valid, maxmetric_valid, tmaxmetric_valid,  fd_list_valid ]
                        ,feed_dict={iter: i}, options=options, run_metadata=run_metadata)

                    #print(c)
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

            f.write("Valid epoch: " + str(i) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                    + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(
                tmmloop) + "\n" + "test epoch end\n")

            f.flush()

            fdloop = 0.0
            mxmloop = 0.0
            mjmloop = 0.0
            tmmloop = 0.0
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
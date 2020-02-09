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
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

def state_activation(x,y):
    #i*x
    #x = tf.complex(x,tf.zeros_like(x))
    cx= tf.complex(0.0,1.0)*tf.complex(x,tf.zeros_like(x))
    #cx= tf.complex(0.0,1.0)*x
    vect = tf.complex(y,tf.zeros_like(y))*tf.math.exp(cx)
    tmp = tf.multiply(tf.conj(vect),vect)
    dem = tf.rsqrt(tf.reduce_sum(tmp,axis=1))
    #dem= 1/(tf.norm(vect,axis=1))
    tmp= tf.einsum('ij,i->ij',vect,dem)
    return tmp

def state_activation_0(x):
    #i*x
    #x = tf.complex(x,tf.zeros_like(x))
    cx= tf.complex(0.0,1.0)*tf.complex(x,tf.zeros_like(x))
    #cx= tf.complex(0.0,1.0)*x
    vect = tf.math.exp(cx)
    tmp = tf.multiply(tf.conj(vect),vect)
    dem = tf.rsqrt(tf.reduce_sum(tmp,axis=1))
    #dem= 1/(tf.norm(vect,axis=1))
    tmp= tf.einsum('ij,i->ij',vect,dem)
    return tmp



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
    valid_batch_size=8
    iterepoch= datasize / batch_size
    itervalid= 5000 / valid_batch_size
    nbqubits= 16
    targetnbqubit=4
    aritycircuitsize= 8
    aritycircuitdepth= 13

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int64, shape=[None])
    #anc = tf.zeros([batch_size,1])
    ancinit=1*tf.ones_like(y)
    anc0 = unionlayer.inttoqubit(ancinit,8)
    #anc1 = unionlayer.inttoqubit(ancinit,4)
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
    # lrdecay =learningrate
    lrdecay= tf.complex(learningrate / tf.add(iter/10 , 1.0),0.0)
    #lrdecay = learningrate*tf.complex(tf.pow(0.5,tf.floor(iter / 2)), 0.0)

    #init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**nbqubits, 2**nbqubits], initializer=init)
    #real = tf.eye(2 ** nbqubits, dtype="float32")
    #imag = tf.zeros_like(real, dtype="float32")
    #param = tf.complex(real, imag)
    init = tf.initializers.he_normal()
    init2 = tf.initializers.glorot_normal()
    init3 = tf.initializers.zeros()

    w0 = tf.get_variable("nnw0",[784,1024],initializer=init)
    w1 = tf.get_variable("nnw1",[1024,1024], initializer=init)
    #wf = tf.get_variable("nnwf", [512,16], initializer=init2)
    #wfr = tf.get_variable("nnwfr", [512, 16], initializer=init2)
    #wfc = tf.get_variable("nnwfc", [512, 16], initializer=init2)
    b0 = tf.get_variable("nnb0", [1024], initializer=init3)
    b1 = tf.get_variable("nnb1", [1024], initializer=init3)
    #bf = tf.get_variable("nnbf", [16], initializer=init3)
    bfr = tf.get_variable("nnbfr", [2**8], initializer=init3)
    #bfc = tf.get_variable("nnbfc", [16], initializer=init3)
    wmlp = tf.get_variable("nnw0f",[1024,2**8],initializer=init2)

    nnparamlist=[]

    nnparamlist.append(w1)
    nnparamlist.append(b1)
    nnparamlist.append(w0)
    nnparamlist.append(b0)
    #nnparamlist.append(bf)
    #nnparamlist.append(wf)
    #nnparamlist.append(bfc)
    #nnparamlist.append(wfc)
    nnparamlist.append(bfr)
    #nnparamlist.append(wfr)
    nnparamlist.append(wmlp)

    #state= state_activation(tf.nn.bias_add(tf.matmul(wf,tf.nn.relu_layer(tf.nn.relu_layer(x,w0,b0),w0,b1)),bf))
    #state=tf.nn.bias_add(tf.matmul(wf,tf.nn.relu_layer(x, w0, b0)),bf)

    l0=tf.nn.relu(tf.matmul(x, w0) +b0 )
    l1=tf.nn.relu(tf.matmul(l0, w1) +b1 )
    #ltr = tf.nn.tanh(tf.matmul(l1, wfr) + bfr)
    #tc = tf.nn.tanh(tf.matmul(l1, wfc) + bfc)
    #lf = state_activation(ltr,ltc)

    #ltr = (tf.matmul(x, wmlp) + bfr)

    ltr = (tf.matmul(l1, wmlp) + bfr)

    lf = state_activation_0(ltr)

    #lf= state_activation(tf.matmul(l1, wf) + bf)
    #lt=tf.sqrt(tf.nn.softmax(tf.matmul(l1, wf) + bf))
    #lf = tf.complex(lt,tf.zeros_like(lt))
    #print(y)
    #print(lf.shape)
    #print(l1)
    #print(lf)
    #print(tf.norm(lf,axis=1))
    #tmp=1 /tf.norm(lf, axis=1)
    #tmp2=tf.einsum('ij,i->ij',lf,tmp)
    #print("lol")
    #print(tmp2)
    #lfn=tf.div(lf,tf.norm(lf,2,axis =1))
    #print(lfn.shape)
    #print(tf.norm(tmp2, axis=1))
    nninput=tf.transpose(lf)
    #sth=unionlayer.join(anc0,nninput )
    #st=unionlayer.join(sth,anc1)
    st=unionlayer.join(anc0,nninput)


    #cir = subcircuit.ArityFillCircuit(nbqubits, 8, 6, "test0", learningrate, momentum)
    #cir = subcircuit.ArityFillCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    #cir = subcircuit.ArityHalfCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    cir = subcircuit.QuincunxCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0")

    #out = cir.forward_two_inputs(nninput,stateinput)
    #out = cir.forward_two_inputs(nninput, nninput)
    #out = cir.forward(nninput)
    out = cir.forward(st,1)
    #for x in range(0,8):

    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)
    #out = cir.recursion(out)

    outvalid = cir.forward_nesterov_test(st,1,True)
    #print(tf.norm(out, axis=0))
    #out = tf.div(out, )
    #print(out.shape)
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
    mcls=lossfunction.maxclass(out, labels,nbqubits,targetnbqubit,16,"inttoqubit")

    tmaxmetric =  lossfunction.max_metric(out, labels,nbqubits,targetnbqubit,16,"inttoqubit")


    fd_list_valid = lossfunction.fidelity_partial_list(outvalid, target,nbqubits,targetnbqubit)
    #cost=tf.reduce_mean(fd_list)
    majmetric_valid = lossfunction.majority_metric(fd_list_valid)
    maxmetric_valid = lossfunction.max_metric(outvalid, labels,nbqubits,targetnbqubit,10,"inttoqubit")
    cost_valid=lossfunction.cross_entropy(outvalid, target,nbqubits,targetnbqubit)
    mcls_valid=lossfunction.maxclass(outvalid, labels,nbqubits,targetnbqubit,16,"inttoqubit")

    tmaxmetric_valid =  lossfunction.max_metric(outvalid, labels,nbqubits,targetnbqubit,16,"inttoqubit")

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
    epoch = 0
    summaries = tf.summary.merge_all()
    f = open("logmnist_hybrid\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))


    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("board\\mnist_hybrid\\\log_" + time.strftime("%Y%m%d-%H%M%S"))

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
            fd,mjm,mxm,tmm,mc,up, s, nnup, wnn,fdlist, nin = sess.run([cost,majmetric,maxmetric,tmaxmetric, mcls,updates, summaries,grads,nnparamlist,fd_list,nninput],feed_dict={x: input_batch,y: labels_batch,iter: epoch}, options=options, run_metadata=run_metadata)
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
                print(" true max accuracy = " + str(tmmloop / dem))
                print(" fdlist")
                print(fdlist)
                print(" max class / label")
                print(mc)
                print(labels_batch)

                #f.write("iter loss list: \n" + str(fd_list)+"\n")
                #f.flush()
            if (i + 1) % 150 == 0 and epoch < 4:
                print("n input")
                print(nin)
                print(np.linalg.norm(nin, ord=2,axis=0))

            if (i + 1) % iterepoch == 0 and i > 0:
                fdloop /= iterepoch
                mjmloop /= iterepoch
                mxmloop /= iterepoch
                tmmloop /= iterepoch
                epoch += 1
                print("epoch " + str(epoch) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))
                summary_writer.add_summary(s, epoch)
                if fdloop < min:
                    min = fdloop
                    iterbest = epoch
                    f.write("iter: " + str(epoch) + "\nfd: " + str(fdloop) +  "\nmajority accuracy = " + str(mjmloop)
                        + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop) + "\nparam: "
                            + str(up) + "\n" + "iter end\n")
                    f.flush()

                fdloop = 0.0
                mxmloop = 0.0
                mjmloop = 0.0
                tmmloop = 0.0

                #input_batch, labels_batch = mnist.validation.images ,mnist.validation.labels
                for  i in range(0,5000//valid_batch_size):
                    input_batch, labels_batch = mnist.validation.next_batch(valid_batch_size)
                    input_batch = input_batch / 255
                    fd_valid, mjm_valid, mxm_valid, tmm_valid, fdlist_valid, = sess.run(
                        [cost_valid, majmetric_valid, maxmetric_valid, tmaxmetric_valid,  fd_list_valid ], feed_dict={x: input_batch, y: labels_batch, iter: epoch}, options=options, run_metadata=run_metadata)

                    fdloop += fd_valid
                    mjmloop += mjm_valid
                    mxmloop += mxm_valid
                    tmmloop += tmm_valid

                fdloop /= itervalid
                mjmloop /= itervalid
                mxmloop /= itervalid
                tmmloop /= itervalid

                print("Valid epoch " + str(epoch) + "\nfd = " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                      + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop))

                f.write("Valid epoch: " + str(epoch) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop)
                        + "\nmax accuracy = " + str(mxmloop) + "\ntrue max accuracy = " + str(tmmloop) + "\n" + "test epoch end\n")

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
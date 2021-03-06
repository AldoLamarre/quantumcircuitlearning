import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction
import iris_data
import time
import vectorencoder
import numpy as np
from tensorflow.python.client import timeline

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels


def preprocess(input,target):
    #ctes=vectorencoder.gennormpluscte(castinput,3,cte=0.01)
    #ancinit = 3 * tf.ones_like(target)
    #ancinit2 = 63*tf.ones_like(target)
    #anc0 = unionlayer.inttoqubit(ancinit, 8)
    #anc1 = unionlayer.inttoqubit(ancinit2, 6)

    ctes = vectorencoder.gencte(input, 2, cte=0.0)
    #print(ctes)
    encodeddata=vectorencoder.encode_vectors(input,2,ctes)
    colvec =tf.transpose(encodeddata)
    #print(colvec.shape)
    c = unionlayer.join(colvec, colvec)
    #v = unionlayer.join(c, c)
    #w = unionlayer.join(v, v)
    #w = unionlayer.join(v, anc0 )
    #w = unionlayer.join(anc1, w)
    #w = unionlayer.join(c ,anc0 )
    return c

if __name__ == "__main__":
    #tf.enable_eager_execution()
    learningrate = -0.01
    momentum = 0.9
    nbqubits= 4
    targetnbqubit=2
    batch_size = 2
    aritycircuitsize = 2
    aritycircuitdepth = 5


    iter = tf.cast(tf.placeholder(tf.int32, shape=()),tf.float32)

    #lrdecay= tf.complex(learningrate / tf.add(iter , 1.0),0.0)
    #lrdecay = learningrate*tf.complex(tf.pow(0.9,tf.floor(iter / 10)), 0.0)
    #lrdecay= tf.complex(learningrate / tf.add(iter/100 , 1.0),0.0)
    first_decay_steps = 75
    lrdecay = tf.complex(tf.train.cosine_decay_restarts(learningrate, iter, first_decay_steps), 0.0)
    #lrdecay = learningrate
    train, test = iris_data.load_data()
    features, labels = train
    dataset = iris_data.train_input_fn(features, labels,batch_size,None)
    dataset= dataset.map(pack_features_vector)
    inputbatch,targetbatch = dataset.make_one_shot_iterator().get_next()


    testset = iris_data.train_input_fn(features, labels, 30, None)
    testset = testset.map(pack_features_vector)
    testinputbatch, testtargetbatch = testset.make_one_shot_iterator().get_next()
    #ci=tf.round(inputbatch*10)
    #print(inputbatch.shape)
    castinput = tf.cast(inputbatch, dtype="float32")



    #init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**nbqubits, 2**nbqubits], initializer=init)
    #real = tf.eye(2 ** nbqubits, dtype="float32")
    #imag = tf.zeros_like(real, dtype="float32")
    #param = tf.complex(real, imag)

    vectorinputs=preprocess(castinput,targetbatch)
    #gate0 = genericQGate.genericQGate(param, nbqubits, nbqubits//2, 0, learningrate, momentum)
    #gate1 = genericQGate.genericQGate(param, nbqubits, nbqubits//2, 0, learningrate, momentum)
    #gate2 = genericQGate.genericQGate(param, nbqubits, nbqubits//2, 7, learningrate, momentum)

    cir = subcircuit.QuincunxCircuit(nbqubits, aritycircuitsize, aritycircuitdepth, "test0", learningrate, momentum)

    #temp = gate0.forward(vectorinputs)
    #temp1 = gate1.forward(temp)
    #temp2 = gate2.forward(temp1)
    out=cir.forward(vectorinputs)
    #out=tf.div(out, tf.norm(out, axis=0))
    #out = cir.forward(out)
    #out = tf.div(out, tf.norm(out, axis=0))
    #out = cir.forward(out)
    #out = tf.div(out, tf.norm(out, axis=0))
    #out = cir.forward(out)
    # Binary classification class 0 vs rest
    #targetbatch=tf.clip_by_value(targetbatch,0,1)
    #testtargetbatch=tf.clip_by_value(testtargetbatch,0,1)

    target = unionlayer.inttoqubit(tf.cast(targetbatch, dtype="float32"), targetnbqubit)
    #targetmsq = unionlayer.onehotqubits(tf.cast(targetbatch, dtype="float32"), nbqubits)
    #cost = lossfunction.fidelity(out, target)

    #print(out.shape)
    fd_list = lossfunction.fidelity_partial_list(out, target,nbqubits,targetnbqubit)

    majmetric_train = lossfunction.majority_metric(fd_list)
    maxmetric_train  = lossfunction.max_metric(out,targetbatch,nbqubits,targetnbqubit,3,"inttoqubit")


    #weight=tf.clip_by_value(fd_list, 0.0, 0.5)
    #weighted_cost = tf.multiply(fd_list,weight)
    #cost=tf.reduce_mean(fd_list)

    cost = lossfunction.cross_entropy(out, target,nbqubits,targetnbqubit)

    #print(cost)
    #cost = lossfunction.fidelity(out, target)
    #print(cost2)
    #costm = lossfunction.msq_real(tf.real(out), tf.real(targetmsq))
    updates = []
    for gates in cir.gatelist:
       updates.append(gates.sgdnesterov(cost,lrdecay,momentum))

    #update=gate0.sgd(cost)
    scost = tf.summary.scalar(name='cost', tensor=cost)
    #scostm = tf.summary.scalar(name='costm', tensor=costm)

    testready= preprocess(tf.cast(testinputbatch, dtype="float32"),tf.cast(testtargetbatch, dtype="float32"))
    #ttemp = gate0.forward(testready)
    #ttemp1 = gate1.forward(ttemp)
    #ttemp2=gate2.forward(ttemp1)
    outtest = cir.forward_nesterov_test(testready,1,False)
    #outtest = cir.forward(testready)
    #outtest = cir.forward(outtest)
    #outtest = cir.forward(outtest)

    targettest = unionlayer.inttoqubit(tf.cast(testtargetbatch, dtype="float32"), targetnbqubit)
    #targettestmsq = unionlayer.onehotqubits(tf.cast(testtargetbatch, dtype="float32"), nbqubits)
    #costtest = lossfunction.fidelity(outtest, targettest)
    fd_test_list = lossfunction.fidelity_partial_list(outtest, targettest,nbqubits,targetnbqubit)
    #costtest = lossfunction.fidelity_partial(outtest, targettest,nbqubits,targetnbqubit)

    majmetric_test = lossfunction.majority_metric(fd_test_list)
    maxmetric_test = lossfunction.max_metric(outtest, testtargetbatch, nbqubits, targetnbqubit, 3, "inttoqubit")

    #costtest = lossfunction.cross_entropy(outtest,targettest,nbqubits, targetnbqubit)
    costtest =  tf.reduce_mean(fd_test_list)

    #costmtest = lossfunction.msq_real(tf.real(outtest), tf.real(targettestmsq))
    #print(tf.shape(out[1]))
    #print(tf.shape(target[1]))
    #print(tf.shape(outtest[1]))
    #print(tf.shape(targettest[1]))


    flag = 1
    max = 10.0
    itermax = 0
    fdloop=0
    mxmloop=0
    mjmloop = 0
    epoch=0
    summaries = tf.summary.merge_all()
    f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    h = open("log\\param_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    h.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("board\\iris\\log_" + time.strftime("%Y%m%d-%H%M%S"))

        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start = tf.global_variables_initializer()
        sess.run(start)
        for i in range(20000):
            # print("iter "+str(i))
            fd,maj,max,up, s = sess.run([cost,majmetric_train,maxmetric_train ,updates, summaries,],feed_dict={iter: epoch },options=options, run_metadata=run_metadata)
            fdloop += fd
            mxmloop += max
            mjmloop += maj
            # print(fd)
            # print(cm)

            if  i == 0:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                g = open("timeline\\iris\\log_timeline_" +str(batch_size)+ "_" + str(nbqubits) + "_" +str(targetnbqubit) + "_"
                         + str(aritycircuitsize) + "_" + str(aritycircuitdepth) +"_"
                         + time.strftime("%Y%m%d-%H%M%S") + ".json", 'x')
                g.write(chrome_trace)
                g.close()

            if (i+1)%75==0 and i>0:
                fdloop /= 75
                mxmloop /= 75
                mjmloop /= 75
                epoch +=1
                print("iter " + str(epoch) + "\nfd = " + str(fdloop))
                print("majority accuracy = " + str(mjmloop))
                print("max accuracy = " + str(mxmloop))

                print(np.sum(np.matrix(up[0]).H * up[0] - np.eye(2**2, dtype=complex)))
                print(np.sum(np.matrix(up[1]).H * up[1] - np.eye(2**2, dtype=complex)))
                f.write("iter: " + str(epoch) + "\nfd: " + str(fdloop) + "\nmajority accuracy = " + str(mjmloop) + "\nmax accuracy = " +
                        str(mxmloop) + "\n" + "iter end\n")


                #if fdloop < max or epoch <= 130:
                if fdloop < max:
                    max = fdloop
                    itermax = epoch
                    #f.write("iter: " + str(epoch) + "\nfd: " + str(fdloop)  + "\nparam: " + str(
                        #up) + "\nmajority accuracy = " + str(mjmloop) + "\nmax accuracy = " +
                            #str(mxmloop) + "\n" + "iter end\n")
                    #fdtest,maj,mx = sess.run([costtest,majmetric_test,maxmetric_test],feed_dict={iter: epoch })
                    #print("iter test " + str(epoch) + "\nfd  test= " + str(fdtest) +
                    #      "\nmajority accuracy test= " + str(maj) + "\nmax accuracy test= " +
                    #        str(mx))
                    #f.write("iter test: " + str(epoch) + "\nfd test: " + str(fdtest) +
                    #        "\nmajority accuracy test= " + str(maj) + "\nmax accuracy test= " +
                    #        str(mx) +"\nparam: " + str(
                    #    up) + "\n" + "iter test end\n")
                #if epoch % 10 == 0 or epoch >= 130:
                if True:
                    fdtest,ttest,itest,otest,fdlist,maj,mx = sess.run([costtest,targettest,
                                                                       testready,outtest,fd_test_list,
                                                                       majmetric_test,maxmetric_test],feed_dict={iter: epoch })
                    print("iter test " + str(epoch) + "\nfd  test= " + str(fdtest) +
                          "\nmajority accuracy test= " + str(maj) + "\nmax accuracy test= " +
                           str(mx))
                    f.write("iter test:" + str(epoch) + "\nfd test: " + str(fdtest) +
                          "\nmajority accuracy test= " + str(maj) + "\nmax accuracy test= " +
                           str(mx) + "\n" + "iter test end\n")

                    #print("input test " + str(itest)+ "\ntarget" + str(ttest) + "\nout  test= " + str(otest) +
                    #      "\nfd_list  test= " + str(fdlist) +"\ntest end\n")
                    #f.write("input test " + str(itest)+ "\ntarget" + str(ttest) + "\nout  test= " + str(otest) +
                    #        "\nfd_list  test= " + str(fdlist) +"\ntest end\n")
                h.write("iter: " + str(epoch) + "\nparam: " + str(up) + "iter end\n")
                f.flush()
                h.flush()
                fdloop = 0
                mxmloop = 0
                mjmloop = 0


        print("max")
        print(max)
        print(itermax)




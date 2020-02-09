import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
import subcircuit
import lossfunction
import numpy as np
from scipy.stats import unitary_group


import time



if __name__ == "__main__":
    #tf.enable_eager_execution()
    batchsize=2

    real = [[0.0, 1.0], [1.0, 0.0]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    negation = tf.complex(real, imag)

    real = [[1.0 , 0.0], [0.0, -1.0]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    phasechange = tf.complex(real, imag)

    real = [[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0], [0.0, 0.0]]
    hadamard = tf.complex(real, imag)

    real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(2**2, dtype="float32"))
    imag = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(2**2, dtype="float32"))
    paramidentiy = tf.complex(real, imag)



    real = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    imag = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    cnot = tf.complex(real, imag)


    init1 = tf.initializers.he_normal(seed=1024)
    init2 = tf.initializers.glorot_normal(seed=1024)
    init3 = tf.initializers.he_uniform(seed=1024)
    init4 = tf.initializers.glorot_uniform(seed=1024)

    real = tf.get_variable("wr",[2*2,2*2],initializer=init4)
    imag = tf.get_variable("wc",[2*2,2*2],initializer=init4)

    param, nothing  = tf.qr(tf.complex(real, imag))

    # param = tf.convert_to_tensor(unitary_group.rvs(2**2), dtype='complex64')
    paramdagger = tf.transpose(param, conjugate=True)





    real = [[1.0, 0.0, 0.0, 0.0]]
    imag = [[0.0, 0.0, 0.0, 0.0]]
    zerozerocte = tf.complex(real, imag)


    #ci=tf.round(inputbatch*10)
    #f0, f1, f2, f3=tf.split(ci,4,axis=1)
    real = [[1 / tf.sqrt(2.0), 0,0, 1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0, 0.0, 0.0]]
    phiplus = tf.transpose( tf.complex(real,imag))

    real = [[1 / tf.sqrt(2.0), 0, 0, 1 / tf.sqrt(2.0)],[1 / tf.sqrt(2.0), 0, 0, 1 / tf.sqrt(2.0)]]
    imag = [[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]
    phiplus2 = tf.transpose(tf.complex(real, imag))

    phipluscte=tf.identity(phiplus)
    for x in range(0,batchsize-1):
        phipluscte= tf.concat([phipluscte,phiplus],axis=1)






    #zerozero = unionlayer.inttoqubit(tf.zeros([batchsize]) ,2)

    #phiplusbatch=tf.multiply(tf.transpose(phiplus),tf.ones([1,1024],dtype="complex64"))

    #init = tf.orthogonal_initializer
    #real = tf.get_variable("v", [2**2, 2**2], initializer=init)
    #real= tf.eye(2**2, dtype="float32")
    #imag = tf.zeros_like(real, dtype="float32")
    #param = tf.complex(real, imag)


    #real = tf.get_variable("w", [2 ** 1, 2 ** 1], initializer=init)
    real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(2 ** 1, dtype="float32"))
    imag = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(2 ** 1, dtype="float32"))
    param1 = tf.complex(real, imag)

    learningrate=-0.01
    momentum=0.01

    iter = tf.cast(tf.placeholder(tf.int32, shape=()), tf.float32)
    #momentum = tf.where(iter % 100 ==0,tf.complex(0.0,0.0),tf.complex(0.0,0.9))
    #momentum = tf.complex(momentum / tf.add(iter / 1000, 1.0), 0.0)




    # lrdecay= tf.complex(learningrate / tf.add(iter , 1.0),0.0)
    # lrdecay = learningrate*tf.complex(tf.pow(0.9,tf.floor(iter / 10)), 0.0)
    lrdecay = tf.complex(learningrate / tf.add(iter / 1000, 1.0), 0.0)
    #lrdecay = tf.where(iter % 5000 ==1,lrdecay,tf.complex(-1.0,0.0))
    #lrdecay = learningrate

    gatecnot = genericQGate.genericQGate(cnot, 2, 2, 0, 0, 0)

    gatecnot1 = genericQGate.genericQGate(cnot, 3, 2, 1, 0, 0)

    gate0 = genericQGate.genericQGate(cnot, 3, 2, 0, learningrate,momentum)

    gatep = genericQGate.genericQGate(param, 3, 2, 0, learningrate, momentum)

    gatep1 = genericQGate.genericQGate(param, 3, 2, 0, learningrate, momentum)

    gateh = genericQGate.genericQGate(hadamard, 3, 1, 0, learningrate, momentum)

    gate1 = genericQGate.genericQGate(negation, 1, 1, 0, learningrate, momentum)

    gate2 = genericQGate.genericQGate(phasechange, 1, 1, 0, learningrate, momentum)

    #for i in range(8):

    real = tf.random_normal([2,batchsize-2],0,1)
    imag = tf.random_normal([2,batchsize-2],0,1)
    #real = [[0.0,  1.0]]
    #imag = [[0.0, 0.0]]




    temp = tf.complex(real, imag)
    vectorrand= tf.div(temp,tf.norm(temp,axis =0))
    vectorrand = tf.concat([tf.transpose(param1),vectorrand],axis=1)


    prep=unionlayer.join(vectorrand,phipluscte)

    #o0 = gateh.forward(gate0.forward(prep))
    o0=gatep1.forward(gatep.forward(prep))


    (o1zero,r0zero),( o1un, r0un) = lazymeasure.lazymeasureboth(o0)

    (o2zz, r1zz), (o2zu, r1zu) = lazymeasure.lazymeasureboth(o1zero)

    (o2uz, r1uz), (o2uu, r1uu) = lazymeasure.lazymeasureboth(o1un)

    o2=tf.concat([o2zz,gate1.forward(o2zu),o2uz,gate2.forward(gate1.forward(o2uu))],axis=1)

    #o2 = tf.stack([o2zz, o2zu, o2uz, o2uu])

    #o1,r0 = lazymeasure.lazymeasure(o0)

    #o2,r1 = lazymeasure.lazymeasure(o1)


    #o3=gate1.forward(unionlayer.join(unionlayer.inttoqubit(r0,1),o2))

    #o4,r2 = lazymeasure.lazymeasure(o3)

    #o3 =  tf.transpose(tf.where(r1  > 0.9,  tf.transpose(gate1.forward(o2)),   tf.transpose(o2)))

    #o4 =  tf.transpose(tf.where(r0 > 0.9,   tf.transpose(gate2.forward(o3)),    tf.transpose(o3)))


    #o5 = gate2.forward(unionlayer.join(unionlayer.inttoqubit(r1, 1), o4))

    #o6, r3 = lazymeasure.lazymeasure(o3)

    out=o2

    #out = o0
    #measure=[r0,r1]
    #measure = r0zero

    #target=vectorrand
    target = tf.concat([vectorrand,vectorrand,vectorrand,vectorrand],axis=1)
    #cost=lossfunction.fidelity(out,target)
    cost = lossfunction.cross_entropy(out, target, 1, 1)
    # print("iter " + str(i))
    # print("\nfd = " + str(cost))
    # print("target\n" + str(target))
    # print("output\n" + str(o4))
    # print("rawo\n" + str(o2))
    # print("measures neg\n" + str(r1))
    # print("measures phase\n" + str(r0))



    #costm = lossfunction.msq(out, target)
    updates=[]
    #for gates in gate2.gatelist:
    updates.append(gatep.sgdnesterov(cost,lrdecay,momentum=momentum))
    updates.append(gatep1.sgdnesterov(cost, lrdecay,momentum=momentum))




    #updatesrms = []
    #updatesrms.append(gatep.rms(costrms, learningrate))
    #updatesrms.append(gatep1.rms(costrms, learningrate))

    #updates.append(gate0.sgd(cost,learningrate))
    #updates.append(gateh.sgd(cost,learningrate))
    #updates = tf.gradients(ys=cost, xs=gate0.param)

    scost = tf.summary.scalar(name='cost', tensor=cost)
    #scostm = tf.summary.scalar(name='costm', tensor=costm)

    par=gatep.normalise()

    flag = 1
    max = 1
    itermax = 0
    summaries = tf.summary.merge_all()
    #f = open("log\\log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "x")
    #f.write("\n" + time.strftime("%Y%m%d-%H%M%S"))
    with tf.Session() as sess:
        summary_writer =tf.summary.FileWriter("board\\teleport\\log_" + time.strftime("%Y%m%d-%H%M%S"))
        start = tf.global_variables_initializer()
        sess.run(start)
        for i in range(3800):
            #print("iter "+str(i))
            fd,up,s,t,o,rawo,h,c=sess.run([cost, updates,summaries,target,out,o2,hadamard,cnot],feed_dict={iter: i })
            #print(fd)
            #print(cm)
            if i % 1 == 0:
                print("iter " + str(i)+"\nfd = "+str(fd))
                hi=np.matrix(np.kron(np.matrix(h),np.identity(2,dtype=complex)))
                cnotnp = np.matrix(c)
                hc=(hi * cnotnp)
                #upm=np.matrix(np.reshape( np.einsum('ij,jk->ik', np.kron(up[1],np.identity(2,dtype=complex)) , up[0]), [4, 4]))
                upm = np.matrix(np.reshape(np.einsum('ij,jk->ik', up[1], up[0]), [4, 4]))
                fbd=np.linalg.norm( upm- hc)
                print("fbd = " + str(fbd))

                #zznp = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
                #zunp = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
                #uznp = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
                #uunp = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
                idd4 = np.identity(4,dtype=complex)
                zznp = np.array([1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)], dtype=complex)
                zunp = np.array([1.0 / np.sqrt(2), 0.0, 0.0, -1.0 / np.sqrt(2)], dtype=complex)
                uznp = np.array([0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], dtype=complex)
                uunp = np.array([0.0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0], dtype=complex)
                bbatch = np.array([[1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)],
                                   [1.0 / np.sqrt(2), 0.0, 0.0, -1.0 / np.sqrt(2)],
                                   [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
                                   [0.0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0]], dtype=complex)
                #np.einsum('ij,ik->ik', upm, idd4)
                print(np.abs(np.einsum('ki,ki->k',np.einsum('ij,kj->ki', upm[1], bbatch), np.einsum('ij,kj->ki', hc, bbatch))))

                #print("target\n" + str(t))
                #print("output\n" + str(o))
                #print("rawo\n" + str(rawo))
                #print("measures\n" + str(m))
            summary_writer.add_summary(s, i)
            if (fd < max):
                max = fd
                itermax = i
                #np.exp(-1 * fd)
                if (np.exp(-1 * fd)> 1 ):
                    print(up)
                    upn=sess.run([par],feed_dict={iter: i //10 })
                    hi = np.matrix(np.kron(np.matrix(h), np.identity(2, dtype=complex)))
                    cnotnp = np.matrix(c)
                    hc = (hi * cnotnp)
                    upnm=np.reshape(np.einsum('ij,jk->ik',  upn[1], upn[0]), [4, 4])
                    #upnm=np.reshape(np.einsum('ij,jk->ik',  np.kron(upn[1],np.identity(2,dtype=complex)), upn[0]), [4, 4])
                    fbd = np.linalg.norm(np.matrix(upnm) - hc)
                    print("fbd = " + str(fbd))
                    print(np.shape(hc))
                    bbatch = np.array([[1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)],
                                       [1.0 / np.sqrt(2), 1.0, 0.0, -1.0 / np.sqrt(2)],
                                       [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0],
                                       [0.0, 1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0]], dtype=complex)
                    zznp = np.array([1.0/np.sqrt(2),0.0,0.0,1.0/np.sqrt(2)],dtype=complex)
                    zunp = np.array([1.0/np.sqrt(2), 0.0, 0.0, -1.0/np.sqrt(2)], dtype=complex)
                    uznp = np.array([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex)
                    uunp = np.array([0.0,1.0/np.sqrt(2), -1.0/np.sqrt(2), 0.0], dtype=complex)
                    print(np.abs(np.einsum('ik,ik->k', np.einsum('ij,kj->ik', upnm, bbatch), np.einsum('ij,kj->ik', hc, bbatch))))


                    #print(upn)
                    #break
                #f.write("iter:" + str(i) + "\nfd: " + str(fd) + "\nmsq: " + str(cm)+ "\nparam: " + str(up) + "\n" + "iter end\n")
            #learningrate = (learningrate / (i/100 + 1.0))

            #if (fd > 1.1*max):
                #upn = sess.run([par])


            #if (fd > 1.15*max and i>100):
                #momentum = 0.0
            #if (fd > 1.30*max and i>100):
                #break
            #if (fd < 1.10*max and momentum < 0.9 ):
                 #momentum += 0.01

        print("max")
        print(max)
        print(itermax)
        #f.write(
            #"iter:" + str(i) + "\nfd: " + str(fd)  + "\nparam: " + str(up) + "\n" + "iter end\n")


    #cir = subcircuit.ArityFillCircuit(10, 2, 10, "test")
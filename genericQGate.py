import tensorflow as tf
#import tensorflow_probability as tfp

class genericQGate:

    gatelist = []

    def __init__(self,param, nbqubitinput,  nbqubitgatesize,  posfirstqubit, learningrate=0,  momentum=0):
        self.momentum = momentum
        self.learningrate = learningrate
        self.posfirstqubit = posfirstqubit
        self.nbqubitGatesize = nbqubitgatesize
        self.nbqubitinput = nbqubitinput
        self.param=tf.get_variable("w"+str(len(self.gatelist)), initializer=param)
        #self.param = tf.get_variable("w", initializer=param)
        self.prev=tf.zeros_like(self.param)
        self.nesterovprev = tf.identity(self.param)
        self.rmsprev=tf.zeros_like(self.param)
        self.gatelist.append(self)
        self.time=1
        self.amsgradprev=tf.real(tf.zeros_like(self.param))


    def forward(self, input):
        self.input = input
        self.sizeF = 1 << self.posfirstqubit
        self.sizeQ = 1 << self.nbqubitGatesize
        self.sizeP = 1 << (self.nbqubitinput - self.posfirstqubit - self.nbqubitGatesize)

        temp=self.tensor(input, self.param, self.sizeF, self.sizeQ, self.sizeP)

        return temp

    def forward_nesterov_switch(self):
        self.prev = self.param
        self.param = self.nesterovprev
        self.nesterovprev = self.prev

        return 0

    def normalise(self):
        self.param, nothing = tf.qr(self.param)

    def normalise_nesterov(self):
        self.forward_nesterov_switch()
        self.normalise()

    def tensor(self, input,matrix,sizeF,sizeQ,sizeP):
        shape=tf.shape(input)


        return tf.reshape(tf.einsum('jc,ickb->ijkb', matrix,
                                    tf.reshape(input, [sizeF, sizeQ, sizeP,shape[1]])), shape)

    def projection(self,derivative,param,learningrate):

        tangent=tf.subtract(tf.matmul(derivative,param,adjoint_a=True),tf.matmul(param,derivative,adjoint_a=True))
        identityreal=tf.eye(tangent.shape[0].value)
        zero=tf.zeros_like(identityreal)
        identity=tf.complex(identityreal,zero)
        identity = self.mround(identity, 1)
        #tangent = self.mround(tangent, 1000)
        cayley=tf.matmul(tf.matrix_inverse(tf.add(identity,tf.scalar_mul(learningrate/2,tangent))),tf.subtract(identity,tf.scalar_mul(learningrate/2,tangent)))
        #cayley=self.mround(cayley, 100000)


        return tf.matmul(cayley,param)

    def update(self, derivative,learningrate):
        return self.param.assign(self.projection(derivative,self.param,learningrate))

    def grad(self,cost):
        return tf.squeeze(tf.gradients(ys=cost, xs=self.param))

    def sgd(self,cost,learningrate,momentum=0.9):
        dW =(1-momentum) *self.grad(cost)+ momentum*self.prev
        self.prev =dW

        return self.update(dW,learningrate)

    # adapted  for ml From arxiv 1903.05204
    # does / should not really work
    def sgdtm(self,cost,learningrate):
        x=self.grad(cost)
        identityreal = tf.eye(x.shape[0].value)
        zero = tf.zeros_like(identityreal)
        identity = tf.complex(identityreal, zero)

        V= 2.0*x * tf.matrix_inverse(identity+tf.matmul(x,self.prev,adjoint_a=True))
        Y = self.projection(V,x,1+self.momentum)
        self.prev =  x

        return self.update(Y,learningrate)

    # adapted  for ml From arxiv 1903.05204
    def sgdnesterov(self,cost,learningrate,momentum):

        X = self.update(self.grad(cost), learningrate)
        # x=self.grad(cost)
        identityreal = tf.eye(X.shape[0].value)
        zero = tf.zeros_like(identityreal)
        identity = tf.complex(identityreal, zero)

        V = 2.0 * X * tf.matrix_inverse(identity + tf.matmul(X, self.nesterovprev, adjoint_a=True))
        self.param = self.projection(V, self.nesterovprev, 1 + self.momentum)
        self.nesterovprev = X

        #V = 2.0 * X * tf.matrix_inverse(identity + tf.matmul(X, self.param, adjoint_a=True))
        #self.nesterovprev = self.projection(V, self.nesterovprev, 1 + self.momentum)
        #self.param = X

        return X

    def __str__(self):
        return "gqgate: num_qubit_input: " +str(self.nbqubitinput) + " num_qubit_gate:"+str(self.nbqubitGatesize)\
        + " pos_first_qubit :" + str(self.posfirstqubit) +"\n"

    def mround(self,m,i):
        m = tf.complex(tf.cast(tf.round(tf.real(m) * i), dtype="float32"),
                        tf.cast(tf.round(tf.imag(m) * i), dtype="float32"))
        return  tf.div(m, i)

    # The optimisers do not work
    def rms(self, cost,learningrate,gamma=0.9):
        dW = self.grad(cost)
        self.rmsprev = gamma *  self.rmsprev  + (1.0-gamma) * tf.square(dW)
        #div=tf.complex(self.rmsprev,(tf.zeros_like(self.rmsprev)))
        dW = dW /(tf.sqrt(self.rmsprev) + 0.00000001)
        #dW=self.mround(dW, 10000)
        return self.update(dW,learningrate)

    def adam(self, cost, beta1=0.9,beta2=0.999,epsilon=0.00000001):
        dW = self.grad(cost)
        m=self.prev=beta1* self.prev +(1- beta1)*dW
        self.rmsprev = beta2 * self.rmsprev + (1 - beta2) * tf.square(tf.abs(dW))
        v = tf.complex(self.rmsprev, (tf.zeros_like(self.rmsprev)))
        mhat=tf.div(m,tf.complex(1-tf.pow(beta1,self.time),0.0))
        vhat = tf.div(v, tf.complex(1 - tf.pow(beta2,self.time),0.0))
        dW = tf.div(mhat, tf.sqrt(vhat) + epsilon)
        self.time+=1
        # dW=self.mround(dW, 10000)
        return self.update(dW)

    def amsgrad(self, cost, beta1=0.9,beta2=0.999,epsilon=0.00000001):
        dW = self.grad(cost)
        m=self.prev=beta1* self.prev +(1- beta1)*dW
        self.rmsprev = beta2 * self.rmsprev + (1 - beta2) * tf.square(tf.abs(dW))
        self.amsgradprev =tf.maximum( self.amsgradprev, self.rmsprev)
        vhat = tf.complex(self.amsgradprev, (tf.zeros_like(self.amsgradprev)))
        dW = tf.div(m, tf.sqrt(vhat) + epsilon)
        #self.time+=1
        # dW=self.mround(dW, 10000)
        return self.update(dW)



    def Hessian(self, cost):
        grad_ys=tf.ones_like(self.param)
        return tf.hessians(ys=cost, xs=self.param)

    def newton(self, cost):

        grad = self.grad(cost)
        grad_y=tf.complex(tf.real(tf.ones_like(grad)),tf.imag(tf.zeros_like(grad)))
        H=tf.squeeze(tf.gradients(ys=grad, xs=self.param, grad_ys= grad_y))
        return  self.update(tf.matrix_inverse(H)*grad)

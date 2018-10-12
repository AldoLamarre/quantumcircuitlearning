import tensorflow as tf


class genericQGate:
    def __init__(self,param, nbqubitinput,  nbqubitgatesize,  posfirstqubit, learningrate=0,  momentum=0):
        self.momentum = momentum
        self.learningrate = learningrate
        self.posfirstqubit = posfirstqubit
        self.nbqubitGatesize = nbqubitgatesize
        self.nbqubitinput = nbqubitinput
        self.param=tf.get_variable("w", initializer=param)
        self.prev=tf.zeros_like(self.param)


    def forward(self, input):
        self.input = input
        self.sizeF = 1 << self.posfirstqubit
        self.sizeQ = 1 << self.nbqubitGatesize
        self.sizeP = 1 << (self.nbqubitinput - self.posfirstqubit - self.nbqubitGatesize)

        temp=self.tensor(input, self.param, self.sizeF, self.sizeQ, self.sizeP)

        return temp

    def tensor(self, input,matrix,sizeF,sizeQ,sizeP):
        shape=input.shape


        return tf.reshape(tf.einsum('jc,ickb->ijkb', matrix,
                                    tf.reshape(input, [sizeF, sizeQ, sizeP,shape[1]])), shape)

    def projection(self,derivative,param,learningrate):

        tangent=tf.subtract(tf.matmul(derivative,param,adjoint_a=True),tf.matmul(param,derivative,adjoint_a=True))
        identityreal=tf.eye(tangent.shape[0].value)
        zero=tf.zeros_like(identityreal)
        identity=tf.complex(identityreal,zero)
        cayley=tf.matmul(tf.matrix_inverse(tf.add(identity,tf.scalar_mul(learningrate/2,tangent))),tf.subtract(identity,tf.scalar_mul(learningrate/2,tangent)))

        return tf.matmul(cayley,param)

    def update(self, derivative):
        return self.param.assign(self.projection(derivative,self.param,self.learningrate))

    def grad(self,cost):
        return tf.squeeze(tf.gradients(ys=cost, xs=self.param))

    def sgd(self,cost):

        dW =(1-self.momentum) *self.grad(cost)+ self.momentum*self.prev
        self.prev =dW

        return self.update(dW)
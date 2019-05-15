import tensorflow as tf
import lazymeasure
import unionlayer
import genericQGate
from abc import ABC, abstractmethod

class SubCircuit:

    @abstractmethod
    def __init__(self,nbqubitinput,name):
        self.nbqubitinput = nbqubitinput
        self.gatelist = []
        self.name=name
    @abstractmethod
    def forward(self,input):
        pass

    @abstractmethod
    def update(self,cost,learningrate):
        pass

#bugger
class ArityFillCircuit(SubCircuit):
    def __init__(self, nbqubitinput,arity,length,name,learningrate=0,  momentum=0):
        super().__init__(nbqubitinput,name)
        init = tf.orthogonal_initializer
        self.arity=arity
        size= 1 << arity
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        param=tf.complex(real,imag)
        for y in range(0,length):
            if(y%2==0):
                for x in range(0,nbqubitinput//arity):
                    with tf.variable_scope(name+"row"+str(y)+"gate"+str(x)):
                        gate = genericQGate.genericQGate(param, nbqubitinput, arity, x*arity,learningrate,momentum)
                        self.gatelist.append(gate)

                leftover=nbqubitinput % arity
                pos = nbqubitinput - leftover
                if(pos <nbqubitinput):
                    with tf.variable_scope(name+"row"+str(y)+"gate"+ str(pos)):
                        size = 1 << leftover
                        #real = tf.get_variable("v",[size,size],initializer=init)
                        real = tf.eye(size, dtype="float32")
                        imag = tf.zeros_like(real)
                        param_l = tf.complex(real, imag)
                        gate = genericQGate.genericQGate(param_l, nbqubitinput, leftover, pos, learningrate, momentum)
                        self.gatelist.append(gate)
            else:
                with tf.variable_scope(name + "row" + str(y) + "gate0"):
                    real = [[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]]
                    imag = [[0.0, 0.0], [0.0, 0.0]]
                    hadamard = tf.complex(real, imag)
                    gate = genericQGate.genericQGate(hadamard, nbqubitinput, 1, 0, learningrate, momentum)
                    self.gatelist.append(gate)

                for x in range(0, (nbqubitinput-1) // arity):
                    with tf.variable_scope(name + "row"+ str(y) + "gate" + str(x+1)):
                        gate = genericQGate.genericQGate(param, nbqubitinput, arity, 1+x * arity, learningrate, momentum)
                        self.gatelist.append(gate)

                leftover = (nbqubitinput-1) % arity
                pos = nbqubitinput - leftover
                if (pos < nbqubitinput):
                    with tf.variable_scope(name + "row" + str(y) + "gate" + str(pos)):
                        size = 1 << leftover
                        #real = tf.get_variable("v",[size,size],initializer=init)
                        real = tf.eye(size, dtype="float32")
                        imag = tf.zeros_like(real)
                        param_l = tf.complex(real, imag)
                        gate = genericQGate.genericQGate(param_l, nbqubitinput, leftover, pos, learningrate, momentum)
                        self.gatelist.append(gate)


    def forward(self,input):
        tmp=input
        for gate in self.gatelist:
            tmp=gate.forward(tmp)

        return tmp


    def update(self,cost,learningrate):
        gradlist=[]
        for gate in self.gatelist:
            gradlist.append(gate.sgd(cost,learningrate))

        return gradlist


    def sgd(self,cost):
        gradlist=[]
        for gate in self.gatelist:
            gradlist.append(gate.sgd(cost))

        return gradlist

    def rms(self, cost,gamma=0.9):
        gradlist = []
        for gate in self.gatelist:
            gradlist.append(gate.rms(cost,gamma))

        return gradlist

    def adam(self, cost,beta1=0.9,beta2=0.999,epsilon=0.00000001):
        gradlist = []
        for gate in self.gatelist:
            gradlist.append(gate.adam(cost,beta1,beta2,epsilon))

        return gradlist

    def amsgrad(self, cost,beta1=0.9,beta2=0.999,epsilon=0.00000001):
        gradlist = []
        for gate in self.gatelist:
            gradlist.append(gate.amsgrad(cost,beta1,beta2,epsilon))

        return gradlist

    def newton(self,cost):
        gradlist=[]
        for gate in self.gatelist:
            gradlist.append(gate.newton(cost))

        return gradlist

if __name__ == "__main__":
    cir=ArityFillCircuit(3,2,2,"test")
    for gate in cir.gatelist:
        print(str(gate))
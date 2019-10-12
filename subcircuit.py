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

#bugger // plus vraiment
class ArityFillCircuit(SubCircuit):
    def __init__(self, nbqubitinput,arity,length,name,learningrate=0,  momentum=0,start=0):
        super().__init__(nbqubitinput,name)
        init = tf.orthogonal_initializer
        self.arity=arity
        size= 1 << arity
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        param=tf.complex(real,imag)
        for y in range(start,length):
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
                        real = tf.get_variable("v",[size,size],initializer=init)
                        #real = tf.eye(size, dtype="float32")
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
                        real = tf.get_variable("v",[size,size],initializer=init)
                        #real = tf.eye(size, dtype="float32")
                        imag = tf.zeros_like(real)
                        param_l = tf.complex(real, imag)
                        gate = genericQGate.genericQGate(param_l, nbqubitinput, leftover, pos, learningrate, momentum)
                        self.gatelist.append(gate)


    def forward(self,input):
        tmp=input
        for gate in self.gatelist:
            tmp=gate.forward(tmp)

        return tmp

    def forward_nesterov_test(self, input):
        for gate in self.gatelist:
            gate.forward_nesterov_switch()
        tmp=self.forward(input)
        for gate in self.gatelist:
            gate.forward_nesterov_switch()


        return tmp


    def update(self,cost,learningrate):
        gradlist=[]
        #for gate in self.gatelist:
            #gradlist.append(gate.sgd(cost,learningrate))
        for x in range(0,len(self.gatelist)):
            gradlist.append(self.gatelist[len(self.gatelist)-1-x].sgd(cost,learningrate))

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



class ArityHalfCircuit(ArityFillCircuit):
    def __init__(self, nbqubitinput,arity,length,name,learningrate=0,  momentum=0):
        super().__init__(nbqubitinput, arity, length, name, learningrate, momentum, start=1)
        init = tf.orthogonal_initializer
        self.arity=arity
        size= 1 << arity
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        param=tf.complex(real,imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param, nbqubitinput//2, arity, 0, learningrate, momentum)
            self.gatelist.insert(0,gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param, nbqubitinput//2, arity, 0, learningrate, momentum)
            self.gatelist.insert(1,gate)




    #
    def forward(self, input):
            half0 = self.gatelist[0].forward(input)
            half1 = self.gatelist[1].forward(input)
            tmp=unionlayer.join(half0,half1)
            for x in range(2,len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)

            return tmp

    def forward_two_inputs(self, input0,input1,start=0):
            half0 = self.gatelist[start].forward(input0)
            half1 = self.gatelist[start+1].forward(input1)
            tmp=unionlayer.join(half0,half1)
            for x in range(start+2,len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)

            return tmp


class ArityHalfAncilaryCircuit(ArityFillCircuit):
    def __init__(self, nbqubitinput,arity,length,name,nbanc, anc, learningrate=0,  momentum=0):
        super().__init__(nbqubitinput, arity, length, name, learningrate, momentum, start=1)
        init = tf.orthogonal_initializer
        self.arity=arity
        size= 1 << (arity-nbanc//2)
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        param=tf.complex(real,imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param, (nbqubitinput-nbanc)//2, arity-nbanc//2, 0, learningrate, momentum)
            self.gatelist.insert(0,gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param, (nbqubitinput-nbanc)//2, arity-nbanc//2, 0, learningrate, momentum)
            self.gatelist.insert(1,gate)
        self.ancilaries = anc

    #def set_ancilaries(self, anc):
       #self.ancilaries = anc


    #
    def forward(self, input):
            half0 = self.gatelist[0].forward(input)
            half1 = self.gatelist[1].forward(input)
            tmp=unionlayer.join(half0,half1)
            tmp=unionlayer.join(tmp,self.ancilaries)
            for x in range(2,len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)

            return tmp

    def forward_two_inputs(self, input0,input1,start=0):
            half0 = self.gatelist[start].forward(input0)
            half1 = self.gatelist[start+1].forward(input1)
            tmp=unionlayer.join(half0,half1)
            for x in range(start+2,len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)

            return tmp


class ArityHalfNSymCircuit(ArityFillCircuit):
    def __init__(self, nbqubitinput0,nbqubitinput1,arity,length,name,learningrate=0,  momentum=0):
        super().__init__(nbqubitinput0+nbqubitinput1, arity, length, name, learningrate, momentum, start=1)
        init = tf.orthogonal_initializer
        self.arity=arity
        size0= 1 << nbqubitinput0
        size1 = 1 << nbqubitinput1
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size0, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size0, dtype="float32"))
        param0=tf.complex(real,imag)
        real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size1, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size1, dtype="float32"))
        param1=tf.complex(real,imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param0, nbqubitinput0, nbqubitinput0, 0, learningrate, momentum)
            self.gatelist.insert(0,gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param1, nbqubitinput1, nbqubitinput1, 0, learningrate, momentum)
            self.gatelist.insert(1,gate)

    def forward_two_inputs(self, input0,input1):
            half0 = self.gatelist[0].forward(input0)
            half1 = self.gatelist[1].forward(input1)
            tmp=unionlayer.join(half0,half1)
            for x in range(2,len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)

            return tmp


class ArityQuarterCircuit(ArityHalfCircuit):
    def __init__(self, nbqubitinput,arity,length,name,learningrate=0,  momentum=0):
        super().__init__(nbqubitinput, arity, length, name, learningrate, momentum)
        init = tf.orthogonal_initializer
        self.arity=min(arity,nbqubitinput//4)
        size= 1 << self.arity
        #real = tf.get_variable(name+"w",[size,size],initializer=init)
        real= tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        imag =  tf.multiply(1.0/tf.sqrt(2.0),tf.eye(size, dtype="float32"))
        param=tf.complex(real,imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param, nbqubitinput//4, self.arity, 0, learningrate, momentum)
            self.gatelist.insert(0,gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param, nbqubitinput//4, self.arity, 0, learningrate, momentum)
            self.gatelist.insert(1,gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(2)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 4, self.arity, 0, learningrate, momentum)
            self.gatelist.insert(2, gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(3)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 4, self.arity, 0, learningrate, momentum)
            self.gatelist.insert(3, gate)




    #
    def forward(self, input):
        quarter0 = self.gatelist[0].forward(input)
        quarter1 = self.gatelist[1].forward(input)
        quarter2 = self.gatelist[2].forward(input)
        quarter3 = self.gatelist[3].forward(input)
        half0 = unionlayer.join(quarter0, quarter1)
        half1 = unionlayer.join(quarter2, quarter3)
        tmp= super().forward_two_inputs(half0,half1,4)

        return tmp

    #def forward_two_inputs(self, input0,input1):
        #half0 = self.gatelist[0].forward(input0)
        #half1 = self.gatelist[1].forward(input1)
        #tmp=unionlayer.join(half0,half1)
        #for x in range(2,len(self.gatelist)):
         #    tmp = self.gatelist[x].forward(tmp)
       #return tmp



class QuincunxCircuit(SubCircuit):
    def __init__(self, nbqubitinput, arity, length, name, learningrate=0, momentum=0, start=0):
        super().__init__(nbqubitinput, name)
        init = tf.orthogonal_initializer
        self.arity = arity
        size = 1 << arity
        # real = tf.get_variable(name+"w",[size,size],initializer=init)
        real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        imag = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        param = tf.complex(real, imag)
        for y in range(start, length):
            if (y % 2 == 0):
                # Same code as arity filled circuits
                for x in range(0,nbqubitinput//arity):
                    with tf.variable_scope(name+"row"+str(y)+"gate"+str(x)):
                        gate = genericQGate.genericQGate(param, nbqubitinput, arity, x*arity,learningrate,momentum)
                        self.gatelist.append(gate)

                leftover=nbqubitinput % arity
                pos = nbqubitinput - leftover
                if(pos <nbqubitinput):
                    with tf.variable_scope(name+"row"+str(y)+"gate"+ str(pos)):
                        size = 1 << leftover
                        real = tf.get_variable("v",[size,size],initializer=init)
                        #real = tf.eye(size, dtype="float32")
                        imag = tf.zeros_like(real)
                        param_l = tf.complex(real, imag)
                        gate = genericQGate.genericQGate(param_l, nbqubitinput, leftover, pos, learningrate, momentum)
                        self.gatelist.append(gate)

            else:
                for x in range(0, (nbqubitinput - arity//2) // arity):
                    with tf.variable_scope(name + "row" + str(y) + "gate" + str(x + arity//2)):
                        gate = genericQGate.genericQGate(param, nbqubitinput, arity, arity//2 + x * arity, learningrate,
                                                         momentum)
                        self.gatelist.append(gate)

    def forward(self,input,recursion=0):
        tmp=input
        for gate in self.gatelist:
            tmp=gate.forward(tmp)

        return tmp




    def forward_nesterov_test(self, input,recursion=1,normalise=False):

        if normalise:
            for gate in self.gatelist:
                gate.normalise_nesterov()
        else:
            for gate in self.gatelist:
                gate.forward_nesterov_switch()

        tmp=self.forward(input,recursion)
        for gate in self.gatelist:
            gate.forward_nesterov_switch()


        return tmp


class HalfQuincunxCircuit(QuincunxCircuit):
    def __init__(self, nbqubitinput, arity, length, name, learningrate=0, momentum=0, start=0):
        super().__init__(nbqubitinput, arity, length, name, learningrate, momentum, start=1)
        init = tf.orthogonal_initializer
        self.arity = arity
        size = 1 << arity
        # real = tf.get_variable(name+"w",[size,size],initializer=init)
        real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        imag = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        param = tf.complex(real, imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 2, arity, 0, learningrate, momentum)
            self.gatelist.insert(0, gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 2, arity, 0, learningrate, momentum)
            self.gatelist.insert(1, gate)



    def forward(self, input, recursion = 1):
        half0 = self.gatelist[0].forward(input)
        half1 = self.gatelist[1].forward(input)
        tmp = unionlayer.join(half0, half1)

        for i in range(recursion):
            for x in range(2, len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)



        return tmp

    def recursion(self, input):
        tmp = input
        for x in range(2, len(self.gatelist)):
            tmp = self.gatelist[x].forward(tmp)

        return tmp

    def forward_two_inputs(self, input0, input1, start=0):
        half0 = self.gatelist[start].forward(input0)
        half1 = self.gatelist[start + 1].forward(input1)
        tmp = unionlayer.join(half0, half1)
        for x in range(start + 2, len(self.gatelist)):
            tmp = self.gatelist[x].forward(tmp)

        return tmp


class ThirdQuincunxCircuit(QuincunxCircuit):
    def __init__(self, nbqubitinput, arity, length, name, learningrate=0, momentum=0, start=0):
        super().__init__(nbqubitinput, arity, length, name, learningrate, momentum, start=1)
        init = tf.orthogonal_initializer
        self.arity = arity
        size = 1 << arity
        # real = tf.get_variable(name+"w",[size,size],initializer=init)
        real = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        imag = tf.multiply(1.0 / tf.sqrt(2.0), tf.eye(size, dtype="float32"))
        param = tf.complex(real, imag)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(0)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 3, arity, 0, learningrate, momentum)
            self.gatelist.insert(0, gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(1)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 3, arity, 0, learningrate, momentum)
            self.gatelist.insert(1, gate)
        with tf.variable_scope(name + "row" + str(0) + "gate" + str(2)):
            gate = genericQGate.genericQGate(param, nbqubitinput // 3, arity, 0, learningrate, momentum)
            self.gatelist.insert(2, gate)



    def forward(self, input, recursion = 0):
        third0 = self.gatelist[0].forward(input)
        third1 = self.gatelist[1].forward(input)
        third2 = self.gatelist[2].forward(input)
        tmp = unionlayer.join(third0, third1)
        tmp = unionlayer.join(tmp, third2)

        for i in range(recursion):
            for x in range(3, len(self.gatelist)):
                tmp = self.gatelist[x].forward(tmp)



        return tmp

    def recursion(self, input):
        tmp = input
        for x in range(3, len(self.gatelist)):
            tmp = self.gatelist[x].forward(tmp)

        return tmp

    #def forward_two_inputs(self, input0, input1, start=0):
        #half0 = self.gatelist[start].forward(input0)
        #half1 = self.gatelist[start + 1].forward(input1)
        #tmp = unionlayer.join(half0, half1)
        #for x in range(start + 2, len(self.gatelist)):
            #tmp = self.gatelist[x].forward(tmp)

        #return tmp


if __name__ == "__main__":
    #cir=ArityFillCircuit(3,2,2,"test")
    #for gate in cir.gatelist:
        #print(str(gate))

    cir = QuincunxCircuit(16, 8, 3, "test")
    for gate in cir.gatelist:
        print(str(gate))
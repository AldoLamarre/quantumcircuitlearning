import tensorflow as tf
import lossfunction
import genericQGate
tf.enable_eager_execution()


def initialtest():
    real = tf.Variable([[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]])
    imag = tf.Variable([[0.0, 0.0], [0.0, 0.0]])
    hadamard = tf.complex(real, imag)

    real = tf.Variable([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    imag = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    vector1 = tf.complex(real, imag)
    vector1=tf.expand_dims(vector1,1)

    gate=genericQGate.genericQGate(hadamard,3,1,0)
    with tf.GradientTape() as g:
        g.watch(gate.param)
        vector2=gate.forward(vector1)
        cost=lossfunction.fidelity(vector2,vector1)

    print(vector2)
    print(cost)
    print(g.gradient(cost, gate.param))

    print("test2")
    real = tf.Variable([[1 / tf.sqrt(2.0), 1 / tf.sqrt(2.0)],[1 / tf.sqrt(2.0), -1 / tf.sqrt(2.0)]])
    imag = tf.Variable([[0.0,0.0 ],[0.0 , 0.0 ]])
    vector3 = tf.complex(real, imag)
    print(vector3)

    gate2 = genericQGate.genericQGate(hadamard, 1, 1, 0,learningrate=1)
    with tf.GradientTape() as g:
        g.watch(gate2.param)
        vector4 = gate2.forward(vector3)
        cost = lossfunction.fidelity(vector4, vector3)

    print(vector4)
    print(cost)
    dW=g.gradient(cost, gate2.param)
    print(dW)
    gate2.update(dW)
    print(gate2.param)

if __name__ == "__main__":
     initialtest()
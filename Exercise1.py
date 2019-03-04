import numpy as np
from tensorflow.keras import backend as K

#1.1

a = K.placeholder(shape=(5))
b = K.placeholder(shape=(5))
c = K.placeholder(shape=(5))

compute_tensor = a*a + b*b + c*c + 2 * b*c

compute = K.function(inputs=[a, b, c], outputs=(compute_tensor,))

# TEST
a1=np.array([1, 2, 3, 4, 5])
a2=np.array([1, 2, 3, 4, 5])
a3=np.array([1, 2, 3, 4, 5])


result = compute((a1, a2, a3))

print("Result", result)




#1.2


def sinh(x): return 0.5 * (np.exp(x) - np.exp(-x))
def cosh(x): return  0.5 * (np.exp(x) + np.exp(-x))
def tanh(x): return sinh(x)/cosh(x)



x = K.placeholder(shape=())

sinh = 0.5 * (K.exp(x) - K.exp(-x))
cosh = 0.5 * (K.exp(x) + K.exp(-x))
tanh = sinh/cosh

tangenshyperbolicus = K.function(inputs=[x],
                          outputs=(tanh))

print(tangenshyperbolicus(1))

tanhgradient = K.gradients(loss=tanh, variables=[x])
grad_function = K.function(inputs=(x), outputs=(tanhgradient))

# #Test
print("tanh(x) x = -100", tangenshyperbolicus(-100))
print("tanh(x) x = -1", tangenshyperbolicus(-1))
print("tanh(x) x = 1", tangenshyperbolicus(1))
print("tanh(x) x = 100", tangenshyperbolicus(100))

print("d/dx tanh(x) x = -100", grad_function(-100))
print("d/dx tanh(x) x = -1", grad_function(-1))
print("d/dx tanh(x) x = 1", grad_function(1))
print("d/dx tanh(x) x = 100", grad_function(100))


#1.3


w_3 = K.ones(shape=(1, 2))
b_3 = K.ones(shape=(1,))
x_3 = K.placeholder(shape=(2,1))

a = K.dot(w_3,x_3) + b_3
function_output = 1/(1+K.exp(-a))

random_function = K.function(inputs=(x_3), outputs=(function_output))

test = np.array([[2.0], [3.0]], dtype="float")
result = (random_function(test))

#Test
print(K.shape(result))
print("result", result)



#1.4
x_3 = K.placeholder(shape=())
n = 5
result = 0

polynom = []
for exponent in range(n + 1):
    factor = K.placeholder(shape=())
    polynom.append(factor)
    result += polynom[-1] * x_3**exponent

grad_polynom = K.gradients(loss=result, variables=polynom)

#Test
print("1.4:")
print(grad_polynom)

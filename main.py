import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import special

def TFFpolyevalfast(poly_coef_tf,zz_tf,poly_order):
    v_tf = poly_coef_tf[0]
    for i in range(1,poly_order[0]):
        v_tf += poly_coef_tf[i]*tf.math.pow(zz_tf,i)
    return v_tf

def TFFgjintegrate(Q,zz_tf,int_weights_tf,gjpower):
    #print(Q*tf.pow(1-zz_tf,-gjpower))
    return tf.matmul(0.5*Q*tf.pow(1-zz_tf,-gjpower),int_weights_tf)

def TFFpolyderiv(poly_coef_tf,zz_tf,poly_order):
    poly_coef_tf_deriv = poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)
    poly_order_new = np.array([poly_order[0]-1,1]).astype(np.int64)
    return TFFpolyevalfast(poly_coef_tf_deriv,zz_tf,poly_order_new)


tf.enable_eager_execution() #keras style of tensorflow


#numerical parameters
n_points = 100
poly_order = 20
gjpower = -0.5

module_testing_on = True


#redefining for tensorflow use
poly_order = np.array([poly_order,1]).astype(np.int64)
poly_index = tf.constant(np.arange(0,poly_order[0],1).reshape((poly_order[0],1)),dtype=tf.float32)



#defining gauss-jacobi grid
gaussjacobi = special.roots_jacobi(n_points,gjpower,0)
zz_tf = tf.constant(gaussjacobi[0],dtype=tf.float32) #grid
int_weights_tf = tf.constant(gaussjacobi[1].reshape((n_points,1)),dtype=tf.float32)

#defining polynomial

if module_testing_on:
    #testing where the velocity field is linear u = z
    poly_coef_np = np.zeros((poly_order[0],1))
    poly_coef_np[1] = 1
    poly_coef_tf = tf.constant(poly_coef_np,dtype=tf.float32)
else:
    poly_coef_tf = tf.random.normal(poly_order)

#IMPORTANT: factor of a half needed in all integrals to convert from [0,1] domain to [-1,1] domain
ubar_tf = tf.matmul(tf.constant(np.ones((1,poly_order[0]),dtype=np.float32)),tf.math.divide(poly_coef_tf,(poly_index+1)))

#velocity rescaled in the [-1:1] domain and normalized
v_tf = TFFpolyevalfast(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)/ubar_tf

#calculating entropy
#H = tf.matmul(tf.math.log(v_tf),int_weights_tf)

test = TFFgjintegrate(v_tf,zz_tf,int_weights_tf,gjpower)

dvdz_tf = TFFpolyderiv(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
H = TFFgjintegrate(tf.math.log(dvdz_tf),zz_tf,int_weights_tf,gjpower)
print(H)
plt.plot(zz_tf.numpy(),v_tf.numpy().flatten())
plt.pause(1000)
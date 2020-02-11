import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import special

def TFpolyevalfast(poly_coef_tf,zz_tf,poly_order):
    v_tf = poly_coef_tf[0]
    for i in range(1,poly_order[0]):
        v_tf += poly_coef_tf[i]*tf.math.pow(zz_tf,i)
    return v_tf

#numerical parameters
n_points = 100
poly_order = 20
gjpower = -0.5


#redefining for tensorflow use
poly_order = np.array([poly_order,1]).astype(np.int64)
poly_index = tf.constant(np.arange(0,20,1),dtype=tf.float32)



#defining gauss-jacobi grid
gaussjacobi = special.roots_jacobi(n_points,0,gjpower)
zz_tf = tf.constant(gaussjacobi[0],dtype=tf.float32) #grid
int_weights_tf = tf.constant(gaussjacobi[1])

#defining polynomial
poly_coef_tf = tf.random.normal(poly_order)


#velocity rescaled in the [-1:1] domain
v_tf = TFpolyevalfast(poly_coef_tf,0.5*zz_tf + 0.5,poly_order) 


#IMPORTANT: factor of a half needed in all integrals to convert from [0,1] domain to [-1,1] domain
ubar_tf = tf.matmul(tf.constant(np.ones((1,poly_order[0]),dtype=np.float32)),tf.math.divide(poly_coef_tf,(poly_index+1)))
print(poly_coef_tf/(poly_index+1))
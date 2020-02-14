import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import special

def TFFpolyevalfast(poly_coef_tf,zz_tf,poly_order):
    v_tf = poly_coef_tf[0]
    for i in range(1,poly_order[0]):
        v_tf += poly_coef_tf[i]*tf.math.pow(zz_tf,i)
    return tf.reshape(v_tf,(tf.shape(zz_tf)[0],1))

def TFFgjintegrate(Q,zz_tf,int_weights_tf,gjpower):
    #print(Q*tf.pow(1-zz_tf,-gjpower))
    #print(Q)
    #print(zz_tf)
    #print(int_weights_tf)
    return tf.matmul(tf.transpose(0.5*Q*tf.pow(1-zz_tf,-gjpower)),int_weights_tf)

def TFFpolyderiv(poly_coef_tf,zz_tf,poly_order):
    poly_coef_tf_deriv = poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)
    poly_order_new = np.array([poly_order[0]-1,1]).astype(np.int64)
    return TFFpolyevalfast(poly_coef_tf_deriv,zz_tf,poly_order_new)


tf.enable_eager_execution() #keras style of tensorflow


#numerical parameters
n_points = 100
poly_order = 20
gjpower = -0.5
chi2 = 1.2

#SGD parameters
SGD_rate = 1e-8
SGD_noise = SGD_rate*1e4

#weighting for lagrange multipliers
LM_boundary_weight = 10000
LM_positive_weight = 100
LM_shape_factor = 1000

#module testing mode:
module_testing_on = True


#redefining for tensorflow use
poly_order = np.array([poly_order,1]).astype(np.int64)
poly_index = tf.constant(np.arange(0,poly_order[0],1).reshape((poly_order[0],1)),dtype=tf.float32)



#defining gauss-jacobi grid
gaussjacobi = special.roots_jacobi(n_points,gjpower,0)
zz_tf = tf.constant(gaussjacobi[0].reshape((n_points,1)),dtype=tf.float32) #grid
int_weights_tf = tf.constant(gaussjacobi[1].reshape((n_points,1)),dtype=tf.float32)
#print(tf.shape(zz_tf).numpy()[0])
#defining polynomial

if module_testing_on:
    #testing where the velocity field is linear u = z
    poly_coef_np = np.zeros((poly_order[0],1))
    poly_coef_np[0] = 1.0
    poly_coef_np[1] = 3.0
    poly_coef_np[2] = -1
    poly_coef_tf = tf.constant(poly_coef_np,dtype=tf.float32)
else:
    poly_coef_tf = tf.random.normal(poly_order)

for i in range(0,300):
    with tf.GradientTape() as g:
        g.watch(poly_coef_tf)
        #IMPORTANT: factor of a half needed in all integrals to convert from [0,1] domain to [-1,1] domain
        ubar_tf = tf.matmul(tf.constant(np.ones((1,poly_order[0]),dtype=np.float32)),tf.math.divide(poly_coef_tf,(poly_index+1)))
        poly_coef_tf /= ubar_tf
        #velocity rescaled in the [-1:1] domain and normalized
        v_tf = TFFpolyevalfast(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)



        #first derivative
        dvdz_tf = TFFpolyderiv(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
        #print(tf.math.log(dvdz_tf))
        #print(int_weights_tf)

        #entropy
        #H = TFFgjintegrate(tf.math.abs(tf.math.log(dvdz_tf))+1e-5,zz_tf,int_weights_tf,gjpower)
        H = TFFgjintegrate(tf.math.log(dvdz_tf),zz_tf,int_weights_tf,gjpower)
        #print(H)

        #lagrange multiplier terms:

        #ensure positivity lagrange multiplier term
        LM_positive_tf = TFFgjintegrate(tf.math.exp(-1000*dvdz_tf),zz_tf,int_weights_tf,gjpower)

        #ensure dudz|z=1 = 0 lagrange multiplier term
        LM_boundary_tf = tf.pow(tf.math.reduce_sum(poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)),2)

        #ensure shape factor lagrange multiplier
        LM_chi_tf = tf.math.pow(TFFgjintegrate(tf.math.pow(v_tf,2),zz_tf,int_weights_tf,gjpower) - chi2,2)

        #cost function
        loss = H + LM_positive_tf*LM_positive_weight + LM_boundary_tf*LM_boundary_weight +LM_chi_tf*LM_shape_factor

        #maximisation
        """ opt = tf.keras.optimizers.SGD(learning_rate=0.1)

        opt_op = opt.minimize(loss, var_list=poly_coef_tf)
        opt_op.run()
        """
        grads = g.gradient(loss,poly_coef_tf)
        test_isnan = np.isnan(grads.numpy())
        
        if test_isnan.any():
            print(grads)
            print(poly_coef_tf)
            plt.pause(10000)
        poly_coef_tf -= grads*SGD_rate
        poly_coef_tf += SGD_noise*tf.math.abs(tf.random.normal(poly_order))*(poly_coef_tf + 1e-1)
        print(loss)
        print(loss.numpy())
        #plt.pause(2)
    del g
    del ubar_tf
    del v_tf
    del dvdz_tf
    del H
    del LM_positive_tf
    del LM_boundary_tf
    del loss
    del grads
    
#print(grads)

#IMPORTANT: factor of a half needed in all integrals to convert from [0,1] domain to [-1,1] domain
ubar_tf = tf.matmul(tf.constant(np.ones((1,poly_order[0]),dtype=np.float32)),tf.math.divide(poly_coef_tf,(poly_index+1)))
poly_coef_tf /= ubar_tf
#velocity rescaled in the [-1:1] domain and normalized
v_tf = TFFpolyevalfast(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
u_bar_test = TFFgjintegrate(v_tf,zz_tf,int_weights_tf,gjpower)
#first derivative
dvdz_tf = TFFpolyderiv(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
#print(tf.math.log(dvdz_tf))
#print(int_weights_tf)

#entropy
#H = TFFgjintegrate(tf.math.abs(tf.math.log(dvdz_tf))+1e-5,zz_tf,int_weights_tf,gjpower)
H = TFFgjintegrate(tf.math.log(dvdz_tf),zz_tf,int_weights_tf,gjpower)
LM_positive_tf = TFFgjintegrate(tf.math.exp(-1000*dvdz_tf),zz_tf,int_weights_tf,gjpower)

#ensure dudz|z=1 = 0 lagrange multiplier term
LM_boundary_tf = tf.pow(tf.math.reduce_sum(poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)),2)

loss = H + LM_positive_tf*LM_positive_weight + LM_boundary_tf*LM_boundary_weight
#print(u_bar_test)
chi2tf = TFFgjintegrate(tf.math.pow(v_tf,2),zz_tf,int_weights_tf,gjpower)
print(H)
print(chi2tf)
""" print(LM_positive_tf*LM_positive_weight)
print(LM_boundary_tf*LM_boundary_weight)
print(loss)
print(poly_coef_tf) """
#plt.plot(zz_tf.numpy(),(tf.math.exp(-10*dvdz_tf)+1).numpy().flatten())
plt.plot(zz_tf.numpy(),(v_tf/ubar_tf).numpy().flatten())
plt.plot(zz_tf.numpy(),1.5*(2*((zz_tf.numpy()+1)/2) - ((zz_tf.numpy()+1)/2)**2))
plt.pause(100000)
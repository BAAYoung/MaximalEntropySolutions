import numpy as np
import matplotlib.pyplot as plt
from scipy import special
#using tensorflow 1 style code
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def TFFpolyevalfast(poly_coef_tf,zz_tf,poly_order):
    v_tf = poly_coef_tf[0]
    for i in range(1,poly_order[0]):
        v_tf += poly_coef_tf[i]*tf.math.pow(1-zz_tf,i)
    return tf.reshape(v_tf,(tf.shape(zz_tf)[0],1))

def TFFgjintegrate(Q,zz_tf,int_weights_tf,gjpower):
    #print(Q*tf.pow(1-zz_tf,-gjpower))
    #print(Q)
    #print(zz_tf)
    #print(int_weights_tf)
    return tf.matmul(tf.transpose(0.5*Q*tf.pow(1-zz_tf,-gjpower)),int_weights_tf)

def TFFpolyderiv(poly_coef_tf,zz_tf,poly_order):
    poly_coef_tf_deriv = -poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)
    poly_order_new = np.array([poly_order[0]-1,1]).astype(np.int64)
    return TFFpolyevalfast(poly_coef_tf_deriv,zz_tf,poly_order_new)

def TFFpolyderiv2(poly_coef_tf,zz_tf,poly_order):
    poly_coef_tf_deriv = poly_coef_tf[2:]*tf.constant(np.arange(2,poly_order[0],1).reshape((poly_order[0]-2,1)),dtype=tf.float32)*tf.constant(np.arange(1,poly_order[0]-1,1).reshape((poly_order[0]-2,1)),dtype=tf.float32)
    poly_order_new = np.array([poly_order[0]-2,1]).astype(np.int64)
    return TFFpolyevalfast(poly_coef_tf_deriv,zz_tf,poly_order_new)

#numerical parameters
n_points = 300
poly_order = 4
gjpower = -0.5
chi2 = 1.3
#SGD parameters
SGD_rate = 1e-8
SGD_noise = SGD_rate*1e4

#weighting for lagrange multipliers
LM_boundary_weight = 10000
LM_positive_weight = 1000
LM_shape_factor = 1000
LM_macroE_weight = 1000

#module testing mode:
module_testing_on = True


#redefining for tensorflow use
poly_order = np.array([poly_order,1]).astype(np.int64)
poly_index = tf.constant(np.arange(0,poly_order[0],1).reshape((poly_order[0],1)),dtype=tf.float32)

#defining gauss-jacobi grid
gaussjacobi = special.roots_jacobi(n_points,gjpower,0)
zz_tf = tf.constant(gaussjacobi[0].reshape((n_points,1)),dtype=tf.float32) #grid
int_weights_tf = tf.constant(gaussjacobi[1].reshape((n_points,1)),dtype=tf.float32)

poly_coef_np = np.zeros((poly_order[0]-2,1))
poly_coef_np[0] = -1
poly_coef_np[1] = 0
#poly_coef_np[2] = 0

poly_coef_trainable = tf.Variable(poly_coef_np,trainable=True,dtype=tf.float32)
poly_coef0 = -tf.reshape(tf.reduce_sum(poly_coef_trainable),[1,1])
poly_coef1 = tf.constant(np.zeros((1,1)),dtype=tf.float32)

poly_coef_tf = tf.concat([poly_coef0,poly_coef1,poly_coef_trainable],axis=0)


ubar_tf = tf.matmul(tf.constant(np.ones((1,poly_order[0]),dtype=np.float32)),tf.math.divide(poly_coef_tf,(poly_index+1)))

v_tf = TFFpolyevalfast(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
dvdz_tf = TFFpolyderiv(poly_coef_tf,0.5*zz_tf + 0.5,poly_order)
ReFr = 2




#lagrange multiplier terms:

#ensure positivity lagrange multiplier term
LM_positive_tf = TFFgjintegrate(tf.math.exp(-1000*dvdz_tf),zz_tf,int_weights_tf,gjpower)

#ensure dudz|z=1 = 0 lagrange multiplier term
LM_boundary_tf = tf.pow(tf.math.reduce_sum(poly_coef_tf[1:]*tf.constant(np.arange(1,poly_order[0],1).reshape((poly_order[0]-1,1)),dtype=tf.float32)),2)

#ensure shape factor lagrange multiplier
LM_chi_tf = tf.math.pow(TFFgjintegrate(tf.math.pow(v_tf,2),zz_tf,int_weights_tf,gjpower)/ubar_tf**2 - chi2,2)

#ensure macro energy conservation
LM_macroE_tf = tf.math.pow(TFFgjintegrate(tf.math.pow(dvdz_tf,2) - ReFr*v_tf,zz_tf,int_weights_tf,gjpower),2)

#entropy
u2_tf = TFFgjintegrate(tf.math.pow(v_tf,2),zz_tf,int_weights_tf,gjpower)
H = TFFgjintegrate(tf.math.log(dvdz_tf),zz_tf,int_weights_tf,gjpower)
loss = -H + LM_macroE_tf*LM_macroE_weight #+ LM_chi_tf*LM_shape_factor
#loss = -u2_tf*0.5
optimizer = tf.train.AdadeltaOptimizer(0.05,rho=0.95,epsilon=1e-8)
train_op = optimizer.minimize(loss)

#denergyds = 


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(loss))
    print(session.run(H))
    #plt.pause(10000)

    for j in range(100):
        for i in range(300):
            session.run(train_op)
        print(session.run(loss))
        #print(session.run(H))

    H_np = session.run(H)
    v_np = session.run(v_tf)
    dvdz_np = session.run(dvdz_tf)
    zz_np = session.run(zz_tf)
    chi = session.run(TFFgjintegrate(tf.math.pow(v_tf,2),zz_tf,int_weights_tf,gjpower))/session.run(ubar_tf)**2
    poly_coef_np = session.run(poly_coef_tf)

print(chi)
print(H_np)
#print(poly_coef_np)
plt.plot(zz_np,(v_np))
zzmod = (zz_np+1)/2
vmax = v_np[-1]
plt.plot(zz_np,vmax*(2*zzmod - zzmod**2))
#plt.plot(zz_np,dvdz_np)
plt.pause(100000)
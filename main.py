import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import special

#numerical parameters
Npoints = 100
gjpower = -0.5

#defining gauss-jacobi grid
gaussjacobi = special.roots_jacobi(Npoints,0,gjpower)
zz_tf = tf.constant(gaussjacobi[0])
weights_tf = tf.constant(gaussjacobi[1])


print(zz_tf)


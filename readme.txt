main.py numerically generates the family of maximum entropy solutions for free surface flows for various shape factors, \chi_2 and generates data for H against \chi_2.

For further information see the letter within this repo.

We use a polynomial interpolation function u(z) = (\sum a_j/(j+1)!)\sum a_k z^k / k! between 0 and 1.
The entropy functional is evaluated numerically using a Gauss-Jacobi quadrature.
The lagrange-multipliers (LM) are evaluated analytically with a term for:
    ensuring the derivative is always non-negative (function is monotonic)
    ensuring the derivative at z = 1 is zero
    selecting a specific value of \chi_2

Tensorflow2.0 is use to minimize the negative Entropy-LM term over the weights a_j.


Subfunctions:

TFgaussjacobi
    numerical integration of a function on a re-scaled gauss jacobi grid

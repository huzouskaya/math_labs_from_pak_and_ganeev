import numpy as np
import pandas as pd
import sympy as sp

def factorial(n):
    if n < 0:
        raise ValueError("Факториал определен только для неотрицательных целых чисел")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def lagrange_interpolation(x, y, x_star, degree):
    n = len(x)
    L = 0.0
    for i in range(degree + 1):
        term = y[i]
        for j in range(degree + 1):
            if j != i:
                term *= (x_star - x[j]) / (x[i] - x[j])
        L += term
    return L

def newton_interpolation(x, y, x_star, degree):
    n = len(x)
    F = np.zeros((n, n))
    F[:,0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            F[i,j] = (F[i+1,j-1] - F[i,j-1]) / (x[i+j] - x[i])
    
    result = F[0,0]
    product = 1.0
    for j in range(1, degree + 1):
        product *= (x_star - x[j-1])
        result += F[0,j] * product
    return result

def compute_omega(x, x_star, k):
    product = 1.0
    for i in range(k + 1):
        product *= (x_star - x[i])
    return product

def compute_derivative(func_sym, order, x_values):
    x_sym = sp.symbols('x')
    f_derivative = func_sym.diff(x_sym, order)
    f_derivative_func = sp.lambdify(x_sym, f_derivative, 'numpy')
    return f_derivative_func(x_values)

def select_closest_points(x, x_star, degree):
    i = np.searchsorted(x, x_star) - 1
    i = max(0, min(i, len(x) - degree - 1))
    return x[i:i+degree+1], range(i, i+degree+1)

def compute_interpolation(func, func_sym, a, b, n, x_star, method, degree):
    '''
    This function performs interpolation of a given function using either Lagrange 
    or Newton method with specified degree, and evaluates its accuracy.
    
    Args:
        func: callable - the function to be interpolated (e.g., lambda x: x**2)
        func_sym: sympy expression - symbolic representation of func
        a: float - left boundary of the interpolation interval
        b: float - right boundary of the interpolation interval
        n: int - number of subintervals (will use n+1 points)
        x_star: float - the point at which to compute interpolation
        method: str - interpolation method ('Lagrange' or 'Newton')
        degree: int - degree of interpolation polynomial (1 for linear, 2 for quadratic, etc.)
        
    Returns:
        None: prints detailed results including:
            - Table of selected points
            - Interpolation result
            - Exact value
            - Error estimation
            - Residual term analysis
            
    Prints:
        Detailed step-by-step results of the interpolation process
    '''
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    
    x_selected, indices = select_closest_points(x, x_star, degree)
    y_selected = y[indices]
    
    print(f"\nValue table (selected {degree+1} closest points to x*={x_star}):")
    table = pd.DataFrame({'x': x_selected, 'y': y_selected})
    print(table)
    
    if method == 'Lagrange':
        L = lagrange_interpolation(x_selected, y_selected, x_star, degree)
    elif method == 'Newton':
        L = newton_interpolation(x_selected, y_selected, x_star, degree)
    else:
        raise ValueError("Method must be either 'Lagrange' or 'Newton'")
    
    exact = func(np.array([x_star]))[0]
    error = abs(L - exact)
    
    print(f"\n{method} interpolation result of degree {degree} at x* = {x_star}:")
    print(f"L_{degree}(x*) = {L}")
    print(f"Exact value f(x*) = {exact}")
    print(f"Absolute error = {error:.2e}")
    
    if degree < len(x_selected) - 1:
        derivative_values = compute_derivative(func_sym, degree+1, x_selected)
        f_derivative_min = np.min(derivative_values)
        f_derivative_max = np.max(derivative_values)
        
        omega = compute_omega(x_selected, x_star, degree)
        R_min = (f_derivative_min * omega) / factorial(degree + 1)
        R_max = (f_derivative_max * omega) / factorial(degree + 1)
        R_actual = L - exact
        
        print(f"\nResidual term R_{degree}(x*) estimation:")
        print(f"Minimum value: {R_min:.2e}")
        print(f"Maximum value: {R_max:.2e}")
        print(f"Actual error: {R_actual:.2e}")
        
        required_precision = 10**(-4) if degree == 1 else 10**(-5)
        if error <= required_precision:
            print(f"\n{method} interpolation of degree {degree} is acceptable (error ≤ {required_precision:.0e})")
        else:
            print(f"\n{method} interpolation of degree {degree} is NOT acceptable (error > {required_precision:.0e})")

if __name__ == "__main__":
    x_sym = sp.symbols('x')
    func = lambda x: x**3 - 2*x**2 + np.exp(-x)
    func_sym = x_sym**3 - 2*x_sym**2 + sp.exp(-x_sym)
    
    a = 0.0
    b = 2.0
    n = 10
    x_star = 1.25
    
    method = 'Lagrange'
    degree = 2
    
    compute_interpolation(func, func_sym, a, b, n, x_star, method, degree)
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

def build_finite_differences_table(y):
    """Строит таблицу конечных разностей"""
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    
    return table

def t_calculator(x, x0, h):
    """Вычисляет параметр t для интерполяционных формул"""
    return (x - x0) / h

def newton_1st(x, y, x_star, degree):
    """Первая интерполяционная формула Ньютона (для интерполирования вперед)"""
    h = x[1] - x[0]
    t = t_calculator(x_star, x[0], h)
    table = build_finite_differences_table(y)
    
    result = table[0, 0]
    product = 1.0
    
    for k in range(1, degree + 1):
        product *= (t - (k - 1)) / k
        result += product * table[0, k]
    
    return result

def newton_2nd(x, y, x_star, degree):
    """Вторая интерполяционная формула Ньютона (для интерполирования назад)"""
    h = x[1] - x[0]
    t = t_calculator(x_star, x[-1], h)
    table = build_finite_differences_table(y)
    
    result = table[-1, 0]
    product = 1.0
    
    for k in range(1, degree + 1):
        product *= (t + (k - 1)) / k
        result += product * table[-k-1, k]
    
    return result

def gauss_1st(x, y, x_star, degree):
    """Первая интерполяционная формула Гаусса"""
    h = x[1] - x[0]
    center = len(x) // 2
    t = t_calculator(x_star, x[center], h)
    table = build_finite_differences_table(y)
    
    result = table[center, 0]
    product = 1.0
    
    for k in range(1, degree + 1):
        if k % 2 == 1:
            term = (t - (k//2)) / k
            idx = center - (k//2)
        else:
            term = (t + (k//2 - 1)) / k
            idx = center - (k//2)
        
        product *= term
        result += product * table[idx, k]
    
    return result

def gauss_2nd(x, y, x_star, degree):
    """Вторая интерполяционная формула Гаусса"""
    h = x[1] - x[0]
    center = len(x) // 2
    t = t_calculator(x_star, x[center], h)
    table = build_finite_differences_table(y)
    
    result = table[center, 0]
    product = 1.0
    
    for k in range(1, degree + 1):
        if k % 2 == 1:
            term = (t + (k//2)) / k
            idx = center - (k//2 + 1)
        else:
            term = (t + (k//2)) / k
            idx = center - (k//2)
        
        product *= term
        result += product * table[idx, k]
    
    return result

def stirling(x, y, x_star, degree):
    """Интерполяционная формула Стирлинга"""
    h = x[1] - x[0]
    center = len(x) // 2
    t = t_calculator(x_star, x[center], h)
    table = build_finite_differences_table(y)
    
    result = table[center, 0]
    t_product = 1.0
    
    for k in range(1, degree + 1):
        if k % 2 == 1:
            mu = 0.5 * (table[center - k//2, k] + table[center - k//2 - 1, k])
            term = t / k
        else:
            mu = table[center - k//2, k]
            term = (t**2 - ((k//2 - 1)**2)) / (k)
        
        t_product *= term
        result += t_product * mu
    
    return result

def bessel(x, y, x_star, degree):
    """Интерполяционная формула Бесселя"""
    h = x[1] - x[0]
    center = len(x) // 2
    t = t_calculator(x_star, x[center], h) - 0.5
    table = build_finite_differences_table(y)
    
    mu = 0.5 * (table[center, 0] + table[center-1, 0])
    result = mu
    t_product = 1.0
    
    for k in range(1, degree + 1):
        if k % 2 == 1:
            delta = table[center - (k//2 + 1), k]
            term = (t - 0.5) / k
        else:
            delta = 0.5 * (table[center - k//2, k] + table[center - k//2 - 1, k])
            term = (t**2 - ((k//2 - 1)**2)) / k
        
        t_product *= term
        result += t_product * delta
    
    return result

def estimate_error(func_sym, x_points, x_star, degree):
    """Оценка погрешности интерполяции"""
    x_sym = sp.symbols('x')
    f_derivative = sp.diff(func_sym, x_sym, degree + 1)
    f_derivative_func = sp.lambdify(x_sym, f_derivative, 'numpy')
    
    omega = 1.0
    for xi in x_points[:degree+1]:
        omega *= (x_star - xi)
    
    derivative_values = f_derivative_func(np.linspace(min(x_points), max(x_points), 100))
    f_derivative_min = np.min(derivative_values)
    f_derivative_max = np.max(derivative_values)
    
    R_min = (f_derivative_min * omega) / factorial(degree + 1)
    R_max = (f_derivative_max * omega) / factorial(degree + 1)
    
    return R_min, R_max

def compute_interpolation(func, func_sym, a, b, n, points_to_interpolate):
    """Основная функция для выполнения интерполяции"""
    x = np.linspace(a, b, n+1)
    y = func(x)
    
    diff_table = build_finite_differences_table(y)
    print("\nТаблица конечных разностей:")
    print(pd.DataFrame(diff_table))
    
    results = []
    
    for point in points_to_interpolate:
        print(f"\nInterpolation point: {point}")
        
        degree = min(5, len(x) - 1)
        
        methods = {
            "Newton first": newton_1st(x, y, point, degree),
            "Newton second": newton_2nd(x, y, point, degree),
            "Gauss first": gauss_1st(x, y, point, degree),
            "Gauss second": gauss_2nd(x, y, point, degree),
            "Stirling": stirling(x, y, point, degree),
            "Bessel": bessel(x, y, point, degree)
        }
        
        exact = func(point)
        R_min, R_max = estimate_error(func_sym, x, point, degree)
        
        for method_name, L in methods.items():
            error = abs(L - exact)
            
            results.append({
                'x*': point,
                'Method': method_name,
                'Interpolated': L,
                'Exact': exact,
                'Error': error,
                'Min R': R_min,
                'Max R': R_max
            })
            
            print(f"\n{method_name}:")
            print(f"L_{degree}(x*) = {L:.6f}")
            print(f"Точное значение = {exact:.6f}")
            print(f"Погрешность = {error:.2e}")
        
        print(f"\nОстаточный член R ∈ [{R_min:.2e}, {R_max:.2e}]")
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    def func(x):
        return np.sin(x) + 0.5 * np.cos(2*x)

    x_sym = sp.symbols('x')
    func_sym = sp.sin(x_sym) + 0.5 * sp.cos(2*x_sym)

    a = 0.0
    b = 2*np.pi
    n = 10
    points_to_interpolate = [1.0, 3.0, 5.0]

    results = compute_interpolation(func, func_sym, a, b, n, points_to_interpolate)

    print("\nTable of results:")
    print(results[['x*', 'Method', 'Interpolated', 'Exact', 'Error']])
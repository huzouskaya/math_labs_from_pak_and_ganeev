import numpy as np
import pandas as pd

def midpoint_rectangle(f, a, b, n):
    """3. Формула центральных прямоугольников"""
    h = (b - a) / n
    return h * sum(f(a + (i + 0.5)*h) for i in range(n))

def simpson(f, a, b, n):
    """5. Формула Симпсона"""
    if n % 2 != 0:
        n += 1  # Делаем n четным
    h = (b - a) / n
    return h/3 * (f(a) + 4*sum(f(a + (2*i-1)*h) for i in range(1, n//2 + 1)) 
                + 2*sum(f(a + 2*i*h) for i in range(1, n//2)) + f(b))

def newton_cotes(f, a, b, n, coeffs):
    """Общая формула Ньютона-Котеса"""
    points = np.linspace(a, b, n+1)
    return (b - a) * sum(c * f(x) for c, x in zip(coeffs, points))

def gauss(f, a, b, n, nodes, weights):
    """Формула Гаусса"""
    t = np.array(nodes)
    x = (b + a)/2 + (b - a)/2 * t
    return (b - a)/2 * sum(w * f(xi) for w, xi in zip(weights, x))

nc_coeffs = {
    1: [1/2, 1/2],
    # 2: [1/6, 4/6, 1/6],
    # 3: [1/8, 3/8, 3/8, 1/8],
    # 4: [7/90, 32/90, 12/90, 32/90, 7/90],
    5: [19/288, 75/288, 50/288, 50/288, 75/288, 19/288]
    # 6: [41/840, 216/840, 27/840, 272/840, 27/840, 216/840, 41/840]
}

gauss_params = {
    1: {'nodes': [0], 'weights': [2]},
    # 2: {'nodes': [-0.577350, 0.577350], 'weights': [1, 1]},
    # 3: {'nodes': [-0.774597, 0, 0.774597], 'weights': [5/9, 8/9, 5/9]},
    # 4: {'nodes': [-0.861136, -0.339981, 0.339981, 0.861136], 
    #     'weights': [0.347855, 0.652145, 0.652145, 0.347855]}
}

def integrate(f, a, b, method, n=10):
    """Обертка для вызова методов интегрирования"""
    if method == 'midpoint':
        return midpoint_rectangle(f, a, b, n)
    elif method == 'simpson':
        return simpson(f, a, b, n)
    elif method.startswith('nc'):
        degree = int(method[2:])
        return newton_cotes(f, a, b, degree, nc_coeffs[degree])
    elif method.startswith('gauss'):
        degree = int(method[5:])
        params = gauss_params[degree]
        return gauss(f, a, b, degree, params['nodes'], params['weights'])

def test_func(x):
    return np.sin(x)

methods = [
    ('3. Центральные прямоуг.', 'midpoint', 10),
    ('5. Симпсон', 'simpson', 10),
    ('7. Ньютон-Котес n=1', 'nc1', None),
    ('11. Ньютон-Котес n=5', 'nc5', None),
    ('13. Гаусс 2 точки', 'gauss2', None),
]

results = []
for name, method, n in methods:
    value = integrate(test_func, 0, np.pi/2, method, n) if n else integrate(test_func, 0, np.pi/2, method)
    error = abs(1 - value)
    results.append({'Метод': name, 'Значение': value, 'Ошибка': error})

df = pd.DataFrame(results)
print(df)
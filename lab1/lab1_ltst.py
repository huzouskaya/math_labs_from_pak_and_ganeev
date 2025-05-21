import numpy as np
import sympy as sp
import pandas as pd
from typing import Callable, Tuple

def lagrange_interpolation(x: np.ndarray, y: np.ndarray, x_star: float, degree: int) -> float:
    """
    Вычисляет значение интерполяционного многочлена Лагранжа в заданной точке.
    
    Parameters:
        x (np.ndarray): Массив узлов интерполяции
        y (np.ndarray): Массив значений функции в узлах
        x_star (float): Точка, в которой вычисляется интерполяция
        degree (int): Степень интерполяционного многочлена
        
    Returns:
        float: Значение интерполяционного многочлена в точке x_star
    """
    n = len(x)
    L = 0.0
    for i in range(degree + 1):
        term = y[i]
        for j in range(degree + 1):
            if j != i:
                term *= (x_star - x[j]) / (x[i] - x[j])
        L += term
    return L

def newton_interpolation(x: np.ndarray, y: np.ndarray, x_star: float, degree: int) -> float:
    """
    Вычисляет значение интерполяционного многочлена Ньютона в заданной точке.
    
    Parameters:
        x (np.ndarray): Массив узлов интерполяции
        y (np.ndarray): Массив значений функции в узлах
        x_star (float): Точка, в которой вычисляется интерполяция
        degree (int): Степень интерполяционного многочлена
        
    Returns:
        float: Значение интерполяционного многочлена в точке x_star
    """
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

def compute_omega(x: np.ndarray, x_star: float, k: int) -> float:
    """
    Вычисляет произведение (x_star - x_0)(x_star - x_1)...(x_star - x_k).
    
    Parameters:
        x (np.ndarray): Массив узлов интерполяции
        x_star (float): Точка интерполяции
        k (int): Количество множителей в произведении
        
    Returns:
        float: Значение произведения разностей
    """
    product = 1.0
    for i in range(k + 1):
        product *= (x_star - x[i])
    return product

def compute_derivative(func_sym: sp.Expr, order: int, x_values: np.ndarray) -> np.ndarray:
    """
    Вычисляет производную заданного порядка символьной функции в заданных точках.
    
    Parameters:
        func_sym (sp.Expr): Символьное выражение функции
        order (int): Порядок производной
        x_values (np.ndarray): Точки, в которых вычисляется производная
        
    Returns:
        np.ndarray: Значения производной в заданных точках
    """
    x_sym = sp.symbols('x')
    f_derivative = func_sym.diff(x_sym, order)
    f_derivative_func = sp.lambdify(x_sym, f_derivative, 'numpy')
    return f_derivative_func(x_values)

def select_closest_points(x: np.ndarray, x_star: float, degree: int) -> Tuple[np.ndarray, range]:
    """
    Выбирает (degree+1) ближайших узлов к точке интерполяции.
    
    Parameters:
        x (np.ndarray): Массив всех узлов интерполяции
        x_star (float): Точка интерполяции
        degree (int): Степень интерполяционного многочлена
        
    Returns:
        Tuple[np.ndarray, range]: 
            - Массив выбранных узлов
            - Диапазон индексов выбранных узлов в исходном массиве
    """
    i = np.searchsorted(x, x_star) - 1
    i = max(0, min(i, len(x) - degree - 1))
    return x[i:i+degree+1], range(i, i+degree+1)

def compute_interpolation(
    func: Callable[[float], float],
    func_sym: sp.Expr,
    a: float,
    b: float,
    n: int,
    x_star: float,
    method: str,
    degree: int
) -> None:
    """
    Основная функция для выполнения и анализа интерполяции.
    
    Parameters:
        func (Callable): Функция для интерполяции (например, lambda x: x**2)
        func_sym (sp.Expr): Символьное представление функции
        a (float): Левая граница интервала интерполяции
        b (float): Правая граница интервала интерполяции
        n (int): Количество подынтервалов (используется n+1 точка)
        x_star (float): Точка, в которой выполняется интерполяция
        method (str): Метод интерполяции ('Lagrange' или 'Newton')
        degree (int): Степень интерполяционного многочлена
        
    Returns:
        Выводит подробные результаты интерполяции
        - Таблицу выбранных точек
        - Результат интерполяции
        - Точное значение
        - Ошибку интерполяции
        - Анализ остаточного члена
        - Оценку точности
        
    Note:
        Функция выполняет полный анализ интерполяции, включая оценку погрешности
        и проверку соответствия требуемой точности.
    """
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
        R_min = (f_derivative_min * omega) / np.prod(np.arange(1, degree+2))
        R_max = (f_derivative_max * omega) / np.prod(np.arange(1, degree+2))
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
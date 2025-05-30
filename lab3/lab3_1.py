import numpy as np
import pandas as pd
import sympy as sp
from typing import Tuple, List, Dict, Callable

class LagrangeDifferentiator:
    """
    Класс для численного дифференцирования таблично заданной функции 
    с использованием интерполяционного многочлена Лагранжа.
    
    Attributes:
        h (float): Шаг между узлами
        x_nodes (np.ndarray): Узлы интерполяции
        y_nodes (np.ndarray): Значения функции в узлах
        n (int): Степень многочлена
        k (int): Порядок производной
    """
    
    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray):
        """
        Args:
            x_nodes: Массив узлов интерполяции (равноотстоящие)
            y_nodes: Значения функции в узлах
        """
        self.h = x_nodes[1] - x_nodes[0]
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.n = len(x_nodes) - 1
        
    def lagrange_derivative(self, x_star: float, k: int = 1) -> float:
        """
        Вычисляет k-ю производную многочлена Лагранжа в точке x_star.
        
        Args:
            x_star: Точка вычисления производной
            k: Порядок производной (1 или 2)
            
        Returns:
            Значение k-й производной
        """
        t = (x_star - self.x_nodes[0]) / self.h
        n = self.n
        
        def l(i: int, t_val: float) -> float:
            term = 1.0
            for j in range(n + 1):
                if j != i:
                    term *= (t_val - j) / (i - j)
            return term
        
        def dl(i: int, t_val: float) -> float:
            total = 0.0
            for m in range(n + 1):
                if m == i:
                    continue
                term = 1.0 / (i - m)
                for j in range(n + 1):
                    if j != i and j != m:
                        term *= (t_val - j) / (i - j)
                total += term
            return total / self.h
        
        def d2l(i: int, t_val: float) -> float:
            total = 0.0
            for m in range(n + 1):
                if m == i:
                    continue
                for p in range(n + 1):
                    if p == i or p == m:
                        continue
                    term = 1.0 / ((i - m) * (i - p))
                    for j in range(n + 1):
                        if j != i and j != m and j != p:
                            term *= (t_val - j) / (i - j)
                    total += term
            return total / (self.h ** 2)
        
        if k == 1:
            return sum(y * dl(i, t) for i, y in enumerate(self.y_nodes))
        elif k == 2:
            return sum(y * d2l(i, t) for i, y in enumerate(self.y_nodes))
        else:
            raise ValueError("Поддерживаются только 1-я и 2-я производные")

    def residual_term(self, x_star: float, k: int, f_derivative: Callable) -> Tuple[float, float]:
        """
        Оценивает остаточный член для k-й производной.
        
        Args:
            x_star: Точка вычисления
            k: Порядок производной
            f_derivative: Функция (n+1)-й производной
            
        Returns:
            Кортеж (min_R, max_R) - минимальное и максимальное значение остаточного члена
        """
        n = self.n
        omega = np.prod([x_star - xi for xi in self.x_nodes])
        
        if k == 1:
            domega = sum(np.prod([x_star - xj for j, xj in enumerate(self.x_nodes) if j != i]) for i in range(n + 1))
        elif k == 2:
            domega = 0.0
            for i in range(n + 1):
                for m in range(i + 1, n + 1):
                    term = 1.0
                    for j in range(n + 1):
                        if j != i and j != m:
                            term *= (x_star - self.x_nodes[j])
                    domega += term
        else:
            raise ValueError("Поддерживаются только 1-я и 2-я производные")
        
        x_values = np.linspace(min(self.x_nodes), max(self.x_nodes), 100)
        deriv_values = f_derivative(x_values)
        f_min, f_max = np.min(deriv_values), np.max(deriv_values)
        temp = domega / np.prod(np.arange(1, n+2))
        R_min = f_min * temp
        R_max = f_max * temp
        
        return R_min, R_max

def run_lab_task(
    func: Callable[[float], float],
    func_sym: sp.Expr,
    a: float,
    b: float,
    n_values: List[int],
    k_values: List[int],
    m_values: Dict[int, List[int]]
) -> None:
    """
    Основная функция для выполнения лабораторной работы.
    
    Args:
        func: Исходная функция
        func_sym: Символьное представление функции
        a: Начало интервала
        b: Конец интервала
        n_values: Список степеней многочлена
        k_values: Список порядков производных
        m_values: Словарь {k: [m]} - номера точек для каждого k
    """
    x_sym = sp.symbols('x')
    
    f_derivatives = {
        1: sp.lambdify(x_sym, func_sym.diff(x_sym, 1)),
        2: sp.lambdify(x_sym, func_sym.diff(x_sym, 2))
    }
    
    f_n1_derivative = sp.lambdify(x_sym, func_sym.diff(x_sym, max(n_values) + 1))
    
    results = []
    
    for k in k_values:
        for n in n_values:
            h = (b - a) / n
            x_nodes = np.array([a + i * h for i in range(n + 1)])
            y_nodes = func(x_nodes)
            
            diff = LagrangeDifferentiator(x_nodes, y_nodes)
            
            for m in m_values.get(k, []):
                x_star = a + m * h
                
                try:
                    L_deriv = diff.lagrange_derivative(x_star, k)
                    exact_deriv = f_derivatives[k](x_star)
                    error = abs(L_deriv - exact_deriv)
                    
                    R_min, R_max = diff.residual_term(x_star, k, f_n1_derivative)
                    R_actual = L_deriv - exact_deriv
                    
                    results.append({
                        'k': k,
                        'n': n,
                        'm': m,
                        'x_m': x_star,
                        'L_deriv': L_deriv,
                        'Exact_deriv': exact_deriv,
                        'Error': error,
                        'R_min': R_min,
                        'R_max': R_max,
                        'R_actual': R_actual,
                        'In_range': R_min <= R_actual <= R_max
                    })
                    
                except ValueError as e:
                    print(f"Ошибка для k={k}, n={n}, m={m}: {str(e)}")
    
    df_results = pd.DataFrame(results)
    
    print("\nРезультаты численного дифференцирования:")
    print(df_results)

if __name__ == "__main__":
    x_sym = sp.symbols('x')
    func = lambda x: np.sin(x) + x**2
    func_sym = sp.sin(x_sym) + x_sym**2
    
    a, b = 0.0, 2.0
    n_values = [3, 4, 5, 6]
    k_values = [1, 2]
    m_values = {
        1: [0, 1, 2, 3, 4, 5, 6],
        2: [0, 1, 2, 3, 4, 5]
    }
    
    run_lab_task(func, func_sym, a, b, n_values, k_values, m_values)

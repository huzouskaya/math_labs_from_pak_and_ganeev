import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def separate_roots(f: callable, a: float, b: float, n_intervals: int = 1000) -> list:
    """
    Отделяет корни уравнения f(x) = 0 на интервале [a, b].
    
    Аргументы:
        f: Функция, корни которой ищем
        a: Левая граница интервала
        b: Правая граница интервала
        n_intervals: Количество подынтервалов для поиска
        
    Возвращает:
        Список кортежей с интервалами, где функция меняет знак
    """
    x = np.linspace(a, b, n_intervals)
    brackets = []
    
    for i in range(len(x)-1):
        if f(x[i]) * f(x[i+1]) <= 0:
            brackets.append((x[i], x[i+1]))
    
    return brackets

def newton_method(f: callable, df: callable, x0: float, eps: float = 1e-6, max_iter: int = 100) -> float:
    """
    Находит корень уравнения f(x) = 0 методом Ньютона.
    
    Аргументы:
        f: Функция
        df: Производная функции
        x0: Начальное приближение
        eps: Точность
        max_iter: Максимальное число итераций
        
    Возвращает:
        Найденный корень
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < eps:
            break
        dfx = df(x)
        if dfx == 0:
            break
        x = x - fx / dfx
    return x

def chord_method(f: callable, a: float, b: float, eps: float = 1e-6, max_iter: int = 100) -> float:
    """
    Находит корень уравнения f(x) = 0 методом хорд.
    
    Аргументы:
        f: Функция
        a, b: Границы интервала
        eps: Точность
        max_iter: Максимальное число итераций
        
    Возвращает:
        Найденный корень
    """
    fa, fb = f(a), f(b)
    for _ in range(max_iter):
        x = a - fa * (b - a) / (fb - fa)
        fx = f(x)
        if abs(fx) < eps:
            break
        if fx * fa < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    return x

def choose_initial_points(f: callable, d2f: callable, a: float, b: float) -> tuple:
    """
    Выбирает начальные точки для комбинированного метода.
    
    Аргументы:
        f: Функция
        d2f: Вторая производная
        a, b: Границы интервала
        
    Возвращает:
        (x_newton, x_chord) - точки для методов Ньютона и хорд
    """
    if f(a) * d2f(a) > 0:
        return a, b
    elif f(b) * d2f(b) > 0:
        return b, a
    else:
        # Если условие не выполняется, используем середину интервала для Ньютона
        return (a + b)/2, b
    
f = lambda x: 0.5 * x**2 - np.cos(2*x)
df = lambda x: x + 2*np.sin(2*x)
d2f = lambda x: 1 + 4*np.cos(2*x)

a, b = -5, 5
eps = 1e-6

intervals = separate_roots(f, a, b)
print(f"Найдены интервалы с корнями: {intervals}")

results = []
for a_i, b_i in intervals:
    x_newton, x_chord = choose_initial_points(f, d2f, a_i, b_i)
    
    root_newton = newton_method(f, df, x_newton, eps)
    root_chord = chord_method(f, a_i, b_i, eps)
    
    results.append({
        "Интервал": f"[{a_i:.3f}, {b_i:.3f}]",
        "Корень (Ньютон)": root_newton,
        "Корень (хорды)": root_chord,
        "Разница": abs(root_newton - root_chord)
    })

results_df = pd.DataFrame(results)
print(results_df)

x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, label="f(x) = 0.5x² - cos(2x)")
plt.axhline(0, color='black', linewidth=0.5)

for _, row in results_df.iterrows():
    plt.scatter(row["Корень (Ньютон)"], 0, color='red', marker='o')
    plt.scatter(row["Корень (хорды)"], 0, color='green', marker='x')

plt.title("График функции и найденные корни")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

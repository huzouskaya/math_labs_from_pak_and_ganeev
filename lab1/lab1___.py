import numpy as np
import pandas as pd
import sympy as sp

def interpolate(func, a: float, b: float, n, x_star) -> float: # method_type: str
    '''
    This function doing an interpolation process using .... 
    
    
    Args:
    func:
    a: float - begin of interval for interpolation ...
    
    Return:
    res : float - interpolated value 
    '''
    # if method_type not in ('Lagrange','Newton'):
    #     print('This method not implemented')
    #     return None
    

    h = (b - a) / n
    x = np.arange(a, a + (n + 1) * h, h)
    y = func(x)

    table = pd.DataFrame({'x': x, 'y = f(x)': y})
    print("Таблица значений:")
    print(table)

    i = np.searchsorted(x, x_star) - 1
    L1_x_star = (y[i] * (x_star - x[i + 1]) / (x[i] - x[i + 1]) + 
                  y[i + 1] * (x_star - x[i]) / (x[i + 1] - x[i]))

    print(f"\nРезультат интерполяции в точке x* = {x_star}: L1(x*) = {L1_x_star}") 

    f_x_star = func(np.array([x_star]))[0]

    error_L1 = abs(L1_x_star - f_x_star)
    print(f"Значение функции в точке x*: f(x*) = {f_x_star}")
    print(f"Погрешность линейной интерполяции: {error_L1}")

    if error_L1 <= 1e-4:
        print("Линейная интерполяция допустима с погрешностью не более 10^(-4).")
    else:
        print("Линейная интерполяция не допустима с погрешностью не более 10^(-4).")

    # Вычисление второй производной
    x_sym = sp.symbols('x')
    f_sym = x_sym - sp.cos(x_sym) 
    f_double_prime = sp.diff(f_sym, x_sym, 2)
    f_double_prime_func = sp.lambdify(x_sym, f_double_prime, 'numpy')

    f_double_prime_min = min(f_double_prime_func(x[i]), f_double_prime_func(x[i + 1]))
    f_double_prime_max = max(f_double_prime_func(x[i]), f_double_prime_func(x[i + 1]))

    omega2 = (x_star - x[i]) * (x_star - x[i + 1])
    R1_x_star_min = f_double_prime_min * omega2 / 2
    R1_x_star_max = f_double_prime_max * omega2 / 2
    R1_x_star = L1_x_star - f_x_star

    print(f"\nОстаточный член R1(x*): {R1_x_star}")
    print(f"Минимальное значение остаточного члена: {R1_x_star_min}")
    print(f"Максимальное значение остаточного члена: {R1_x_star_max}")

    if R1_x_star_min < R1_x_star < R1_x_star_max:
        print("Неравенство min R1 < R1(x*) < max R1 выполняется.")
    else:
        print("Неравенство min R1 < R1(x*) < max R1 не выполняется.")

    if i > 0 and i < n - 1:
        L2_x_star = (y[i - 1] * (x_star - x[i]) * (x_star - x[i + 1]) / ((x[i - 1] - x[i]) * (x[i - 1] - x[i + 1])) +
                      y[i] * (x_star - x[i - 1]) * (x_star - x[i + 1]) / ((x[i] - x[i - 1]) * (x[i] - x[i + 1])) +
                      y[i + 1] * (x_star - x[i - 1]) * (x_star - x[i]) / ((x[i + 1] - x[i - 1]) * (x[i + 1] - x[i])))

        error_L2 = abs(L2_x_star - f_x_star)
        print(f"\nРезультат квадратичной интерполяции (Лагранжа) в точке x* = {x_star}: L2(x*) = {L2_x_star}")
        print(f"Погрешность квадратичной интерполяции: {error_L2}")

        if error_L2 <= 1e-5:
            print("Квадратичная интерполяция допустима с погрешностью не более 10^(-5).")
        else:
            print("Квадратичная интерполяция не допустима с погрешностью не более 10^(-5).")

        # Вычисление третьей производной
        f_triple_prime = sp.diff(f_sym, x_sym, 3)
        f_triple_prime_func = sp.lambdify(x_sym, f_triple_prime, 'numpy')

        f_triple_prime_min = min(f_triple_prime_func(x[i - 1]), f_triple_prime_func(x[i]), f_triple_prime_func(x[i + 1]))
        f_triple_prime_max = max(f_triple_prime_func(x[i - 1]), f_triple_prime_func(x[i]), f_triple_prime_func(x[i + 1]))

        omega3 = (x_star - x[i - 1]) * (x_star - x[i]) * (x_star - x[i + 1])
        R2_x_star_min = f_triple_prime_min * omega3 / 6
        R2_x_star_max = f_triple_prime_max * omega3 / 6
        R2_x_star = L2_x_star - f_x_star

        print(f"\nОстаточный член R2(x*): {R2_x_star}")
        print(f"Минимальное значение остаточного члена: {R2_x_star_min}")
        print(f"Максимальное значение остаточного члена: {R2_x_star_max}")

        if R2_x_star_min < R2_x_star < R2_x_star_max:
            print("Неравенство min R2 < R2(x*) < max R2 выполняется.")
        else:
            print("Неравенство min R2 < R2(x*) < max R2 не выполняется.")

        # Вычисление разделенных разностей для интерполяции Ньютона
        divided_differences = np.zeros((3, 3))
        divided_differences[0, 0] = y[i - 1]
        divided_differences[1, 0] = y[i]
        divided_differences[2, 0] = y[i + 1]

        divided_differences[1, 1] = (divided_differences[1, 0] - divided_differences[0, 0]) / (x[i] - x[i - 1])
        divided_differences[2, 1] = (divided_differences[2, 0] - divided_differences[1, 0]) / (x[i + 1] - x[i])
        
        divided_differences[2, 2] = (divided_differences[2, 1] - divided_differences[1, 1]) / (x[i + 1] - x[i - 1])

        L2_x_star_newton = (divided_differences[1, 0] + 
                            divided_differences[1, 1] * (x_star - x[i]) + 
                            divided_differences[2, 2] * (x_star - x[i - 1]) * (x_star - x[i]))

        error_L2_newton = abs(L2_x_star_newton - f_x_star)
        print(f"\nРезультат квадратичной интерполяции (Ньютона) в точке x* = {x_star}: L2(x*) = {L2_x_star_newton}")
        print(f"Погрешность квадратичной интерполяции (Ньютона): {error_L2_newton}")

        if error_L2_newton <= 1e-5:
            print("Квадратичная интерполяция (Ньютона) допустима с погрешностью не более 10^(-5).")
        else:
            print("Квадратичная интерполяция (Ньютона) не допустима с погрешностью не более 10^(-5).")

        print(f"\nСравнение результатов:")
        print(f"L2(x*) (Лагранжа) = {L2_x_star}, L2(x*) (Ньютона) = {L2_x_star_newton}")

    else:
        print("Недостаточно данных для выполнения квадратичной интерполяции.")


if __name__ == "__main__":
    x_sym = sp.symbols('x')
    func = lambda x: x**2 + (-0.5) * np.exp(-x)

    a = 0.1 
    b = 0.6 
    n = 10 
    x_star = 0.37

    interpolate(func, a, b, n, x_star)

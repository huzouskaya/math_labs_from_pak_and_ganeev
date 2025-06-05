import numpy as np
from typing import Tuple, Optional

def monotone_sweep_solver(
    lower_diag: np.ndarray,
    main_diag: np.ndarray,
    upper_diag: np.ndarray,
    f_vector: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Реализация метода монотонной прогонки

    Параметры:
    ----------
    lower_diag : np.ndarray
        Нижняя диагональ (элементы a_i, i=2..n). Должна иметь длину n-1.
    main_diag : np.ndarray
        Главная диагональ (элементы b_i, i=1..n). Должна иметь длину n.
    upper_diag : np.ndarray
        Верхняя диагональ (элементы c_i, i=1..n-1). Должна иметь длину n-1.
    f_vector : np.ndarray
        Вектор правой части (элементы f_i, i=1..n). Должна иметь длину n.

    Возвращает:
    -----------
    Tuple[Optional[np.ndarray], Optional[str]]
        - Если решение найдено: (solution, None)
        - Если ошибка: (None, error_message)
    """
    
    n = len(main_diag)
    if len(lower_diag) != n - 1 or len(upper_diag) != n - 1 or len(f_vector) != n:
        return None, "Несоответствие размеров входных данных"
    
    if n == 0:
        return None, "Пустая система"
    
    alpha = np.zeros(n)
    beta = np.zeros(n)
    solution = np.zeros(n)

    try:
        alpha[1] = -upper_diag[0] / main_diag[0]
        beta[1] = f_vector[0] / main_diag[0]
        
        for k in range(1, n - 1):
            denominator = main_diag[k] + lower_diag[k - 1] * alpha[k]
            if abs(denominator) < 1e-10:
                return None, "Матрица системы вырождена"
            
            alpha[k + 1] = -upper_diag[k] / denominator
            beta[k + 1] = (f_vector[k] - lower_diag[k - 1] * beta[k]) / denominator

        denominator = main_diag[-1] + lower_diag[-1] * alpha[-1]
        if abs(denominator) < 1e-10:
            return None, "Матрица системы вырождена"
        
        solution[-1] = (f_vector[-1] - lower_diag[-1] * beta[-1]) / denominator
        
        for k in range(n - 2, -1, -1):
            solution[k] = alpha[k + 1] * solution[k + 1] + beta[k + 1]

        return solution, None

    except Exception as e:
        return None, f"Ошибка вычислений: {str(e)}"

def main():
    print("МЕТОД МОНОТОННОЙ ПРОГОНКИ")
    print("="*40)
    
    A = np.array([-10, -4, 6, 5, 4])        # Нижняя диагональ
    B = np.array([20, -5, 5, 15, 25, 18])   # Главная диагональ
    C = np.array([-1, -1, 10, -10, 9])      # Верхняя диагональ
    F = np.array([10, -10, 10, 30, 19, 27]) # Вектор b

    solution, error = monotone_sweep_solver(A, B, C, F)
    
    print("\nВходные данные:")
    print(f"A (нижн.): {A}\nB (глав.): {B}\nC (верх.): {C}\nF (вектор b): {F}")
    
    if error:
        print(f"\nОшибка: {error}")
    else:
        print(f"\nРешение: {solution}")

if __name__ == "__main__":
    main()

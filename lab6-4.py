import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


class CubicSpline:
    """
    Класс для построения кубического сплайна с различными граничными условиями
    Поддерживает 4 типа граничных условий
    """
    
    def __init__(self, x: list, y: list, boundary_type: int = 1, boundary_values: tuple = (0, 0)):
        """
        Инициализация сплайна
        
        Параметры:
            x - список узлов интерполяции (должны быть строго возрастающими)
            y - значения функции в узлах
            boundary_type - тип граничных условий (1-4)
            boundary_values - значения производных на границах (для типов 1 и 2)
        """
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.boundary_type = boundary_type
        self.boundary_values = boundary_values
        self.n = len(x) - 1  # Количество интервалов
        self.h = np.diff(self.x)  # Расстояния между узлами
        
        # Вычисление наклонов в зависимости от типа граничных условий
        if boundary_type == 1:
            self.m = self._compute_type1()
        elif boundary_type == 2:
            self.m = self._compute_type2()
        elif boundary_type == 3:
            self.m = self._compute_type3()
        elif boundary_type == 4:
            self.m = self._compute_type4()
        else:
            raise ValueError("Тип граничных условий должен быть 1, 2, 3 или 4")

    def _compute_type1(self) -> np.ndarray:
        """Вычисление наклонов для граничных условий 1-го типа (заданы первые производные)"""
        A = np.zeros((self.n + 1, self.n + 1))
        B = np.zeros(self.n + 1)
        
        # Граничные условия
        A[0, 0] = 1
        B[0] = self.boundary_values[0]
        A[-1, -1] = 1
        B[-1] = self.boundary_values[1]
        
        # Уравнения для внутренних узлов
        for i in range(1, self.n):
            mu = self.h[i-1] / (self.h[i-1] + self.h[i])
            lam = 1 - mu
            A[i, i-1] = mu
            A[i, i] = 2
            A[i, i+1] = lam
            B[i] = 3 * (lam*(self.y[i+1]-self.y[i])/self.h[i] + mu*(self.y[i]-self.y[i-1])/self.h[i-1])
        
        return np.linalg.solve(A, B)

    def _compute_type2(self) -> np.ndarray:
        """Вычисление наклонов для граничных условий 2-го типа (заданы вторые производные)"""
        A = np.zeros((self.n + 1, self.n + 1))
        B = np.zeros(self.n + 1)
        
        # Граничные условия
        A[0, 0] = 2
        A[0, 1] = 1
        B[0] = 3*(self.y[1]-self.y[0])/self.h[0] - self.h[0]*self.boundary_values[0]/2
        
        A[-1, -2] = 1
        A[-1, -1] = 2
        B[-1] = 3*(self.y[-1]-self.y[-2])/self.h[-1] + self.h[-1]*self.boundary_values[1]/2
        
        # Уравнения для внутренних узлов
        for i in range(1, self.n):
            mu = self.h[i-1] / (self.h[i-1] + self.h[i])
            lam = 1 - mu
            A[i, i-1] = mu
            A[i, i] = 2
            A[i, i+1] = lam
            B[i] = 3 * (lam*(self.y[i+1]-self.y[i])/self.h[i] + mu*(self.y[i]-self.y[i-1])/self.h[i-1])
        
        return np.linalg.solve(A, B)

    def _compute_type3(self) -> np.ndarray:
        """Вычисление наклонов для периодических граничных условий (3-й тип)"""
        A = np.zeros((self.n, self.n))
        B = np.zeros(self.n)
        
        # Периодические условия
        for i in range(self.n):
            mu = self.h[i-1]/(self.h[i-1]+self.h[i]) if i > 0 else self.h[-1]/(self.h[-1]+self.h[0])
            lam = 1 - mu
            prev_idx = i-1 if i > 0 else self.n-1
            next_idx = i+1 if i < self.n-1 else 0
            
            A[i, prev_idx] = mu
            A[i, i] = 2
            A[i, next_idx] = lam
            
            dy_prev = self.y[i]-self.y[prev_idx] if i > 0 else self.y[0]-self.y[-1]
            dy_next = self.y[next_idx]-self.y[i] if i < self.n-1 else self.y[0]-self.y[-1]
            h_prev = self.h[i-1] if i > 0 else self.h[-1]
            h_next = self.h[i] if i < self.n-1 else self.h[0]
            
            B[i] = 3 * (lam*dy_next/h_next + mu*dy_prev/h_prev)
        
        m = np.zeros(self.n + 1)
        m[:-1] = np.linalg.solve(A, B)
        m[-1] = m[0]  # Периодическое условие
        
        return m

    def _compute_type4(self) -> np.ndarray:
        """Вычисление наклонов для граничных условий 4-го типа (непрерывность 3-й производной)"""
        A = np.zeros((self.n + 1, self.n + 1))
        B = np.zeros(self.n + 1)
        
        # Первое специальное уравнение
        gamma1 = self.h[0] / self.h[1]
        A[0, 0] = 1
        A[0, 1] = 1 - gamma1**2
        A[0, 2] = -gamma1**2
        B[0] = 2*(self.y[1]-self.y[0])/self.h[0] - 2*gamma1**2*(self.y[2]-self.y[1])/self.h[1]
        
        # Второе уравнение из системы
        mu1 = self.h[1] / (self.h[0] + self.h[1])
        lam1 = 1 - mu1
        A[1, 0] = mu1
        A[1, 1] = 2
        A[1, 2] = lam1
        B[1] = 3*(lam1*(self.y[2]-self.y[1])/self.h[1] + mu1*(self.y[1]-self.y[0])/self.h[0])
        
        # Уравнения для внутренних узлов
        for i in range(2, self.n - 1):
            mu = self.h[i] / (self.h[i-1] + self.h[i])
            lam = 1 - mu
            A[i, i-1] = mu
            A[i, i] = 2
            A[i, i+1] = lam
            B[i] = 3 * (lam*(self.y[i+1]-self.y[i])/self.h[i] + mu*(self.y[i]-self.y[i-1])/self.h[i-1])
        
        # Последнее специальное уравнение
        gammaN = self.h[-1] / self.h[-2]
        A[-2, -3] = gammaN**2
        A[-2, -2] = -(1 - gammaN**2)
        A[-2, -1] = -1
        B[-2] = 2*(gammaN**2*(self.y[-2]-self.y[-3])/self.h[-2] - (self.y[-1]-self.y[-2])/self.h[-1])
        
        # Решение системы
        m = np.zeros(self.n + 1)
        m[:-1] = np.linalg.solve(A[:-1, :-1], B[:-1])
        m[-1] = (B[-2] - A[-2, -3]*m[-3] - A[-2, -2]*m[-2]) / A[-2, -1]
        
        return m

    def __call__(self, x: float) -> float:
        """Вычисление значения сплайна в точке x"""
        i = np.clip(np.searchsorted(self.x, x) - 1, 0, self.n - 1)
        dx = x - self.x[i]
        
        a_i = (6/self.h[i]) * ((self.y[i+1]-self.y[i])/self.h[i] - (self.m[i+1]+2*self.m[i])/3)
        b_i = (12/self.h[i]**2) * ((self.m[i+1]+self.m[i])/2 - (self.y[i+1]-self.y[i])/self.h[i])
        
        return self.y[i] + self.m[i]*dx + a_i*dx**2/2 + b_i*dx**3/6

    def derivative(self, x: float, order: int = 1) -> float:
        """Вычисление производной сплайна в точке x"""
        i = np.clip(np.searchsorted(self.x, x) - 1, 0, self.n - 1)
        dx = x - self.x[i]
        
        a_i = (6/self.h[i]) * ((self.y[i+1]-self.y[i])/self.h[i] - (self.m[i+1]+2*self.m[i])/3)
        b_i = (12/self.h[i]**2) * ((self.m[i+1]+self.m[i])/2 - (self.y[i+1]-self.y[i])/self.h[i])
        
        if order == 1:
            return self.m[i] + a_i*dx + b_i*dx**2/2
        elif order == 2:
            return a_i + b_i*dx
        else:
            raise ValueError("Порядок производной должен быть 1 или 2")

    def integral(self, a: float, b: float) -> float:
        """Вычисление определенного интеграла от a до b"""
        total = 0.0
        start = np.clip(np.searchsorted(self.x, a) - 1, 0, self.n - 1)
        end = np.clip(np.searchsorted(self.x, b) - 1, 0, self.n - 1)
        
        # Добавляем частичные интегралы для каждого отрезка
        for i in range(start, end + 1):
            x_start = max(a, self.x[i])
            x_end = min(b, self.x[i+1]) if i < self.n else b
            if x_start >= x_end:
                continue
                
            dx1 = x_start - self.x[i]
            dx2 = x_end - self.x[i]
            
            a_i = (6/self.h[i]) * ((self.y[i+1]-self.y[i])/self.h[i] - (self.m[i+1]+2*self.m[i])/3)
            b_i = (12/self.h[i]**2) * ((self.m[i+1]+self.m[i])/2 - (self.y[i+1]-self.y[i])/self.h[i])
            
            integral = (self.y[i]*(dx2-dx1) + self.m[i]*(dx2**2-dx1**2)/2 + 
                        a_i*(dx2**3-dx1**3)/6 + b_i*(dx2**4-dx1**4)/24)
            total += integral
        
        return total

    def plot(self, true_func=None, title="Кубический сплайн"):
        """Визуализация сплайна"""
        xs = np.linspace(self.x[0], self.x[-1], 500)
        ys = [self(x) for x in xs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label="Сплайн", color="green")
        plt.plot(self.x, self.y, 'o', label="Узлы интерполяции", color="red")
        
        if true_func:
            true_ys = [true_func(x) for x in xs]
            plt.plot(xs, true_ys, '--', label="Исходная функция", color="yellow")
        
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()


def auto_derivatives(func_expr, x_val):
    """Автоматическое вычисление производных с помощью SymPy"""
    x = sp.symbols('x')
    f = func_expr
    
    # Вычисляем первую и вторую производные
    f1 = sp.diff(f, x)
    f2 = sp.diff(f1, x)
    
    # Преобразуем в числовые функции
    f_num = sp.lambdify(x, f, 'numpy')
    f1_num = sp.lambdify(x, f1, 'numpy')
    f2_num = sp.lambdify(x, f2, 'numpy')
    
    # Вычисляем значения в точке
    return float(f1_num(x_val)), float(f2_num(x_val))


def test_spline():
    """Тестирование сплайна на выбранной функции"""
    x = sp.symbols('x')
    func_expr = sp.sin(x) + sp.cos(2*x)  # Истинная функция
    a, b = 0, 2*np.pi                    # Интервал
    n = 10                               # Количество узлов интерполяции
    boundary_type = 1                    # Тип граничных условий (1-4)
    
    # Автоматическое вычисление производных на границах
    f1_a, f2_a = auto_derivatives(func_expr, a)
    f1_b, f2_b = auto_derivatives(func_expr, b)
    
    if boundary_type == 1:
        boundary_values = (f1_a, f1_b)
        print(f"Первые производные на границах: f'({a})={f1_a:.3f}, f'({b})={f1_b:.3f}")
    elif boundary_type == 2:
        boundary_values = (f2_a, f2_b)
        print(f"Вторые производные на границах: f''({a})={f2_a:.3f}, f''({b})={f2_b:.3f}")
    else:
        boundary_values = (0, 0)
    
    # Числовая функция для вычисления значений
    f_num = sp.lambdify(x, func_expr, 'numpy')
    
    # Построение сплайна
    x_nodes = np.linspace(a, b, n)
    y_nodes = f_num(x_nodes)
    
    spline = CubicSpline(x_nodes, y_nodes, boundary_type, boundary_values)
    
    # Тестирование в случайной точке
    test_x = np.random.uniform(a + 0.1*(b-a), b - 0.1*(b-a))
    print(f"\nТестирование в точке x = {test_x:.3f}:")
    print(f"Истинное значение: {f_num(test_x):.6f}")
    print(f"Значение сплайна:  {spline(test_x):.6f}")
    print(f"Ошибка:            {abs(f_num(test_x) - spline(test_x)):.2e}")
    
    spline.plot(f_num, title=f"Кубический сплайн (тип {boundary_type}) для функции {func_expr}")


if __name__ == "__main__":
    test_spline()
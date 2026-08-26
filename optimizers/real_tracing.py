"""
Модуль для реальной (непараксиальной) трассировки лучей через гармонические линзы.
Учитывает сферическую аберрацию и лучи, идущие на разных расстояниях от оптической оси.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from data.dataset_helper import DataSetHelper
from data.datasets import DATA_SET_2, DATA_SET_3
from utils.unit import UnitType
from utils.keys import DataSetKeys


class RealRayTracer:
    """Класс для реальной трассировки лучей через оптическую систему"""
    
    def __init__(self, dataset: Dict[DataSetKeys, any]):
        self.dataset = dataset
        self.count_linse = dataset['count_linse']
        
    def n_bk7(self, lambda_um: float) -> float:
        """Расчёт показателя преломления BK7 стекла по формуле Зельмейера"""
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
        l2 = lambda_um**2
        n2 = 1 + (B1 * l2) / (l2 - C1) + (B2 * l2) / (l2 - C2) + (B3 * l2) / (l2 - C3)
        return n2**0.5
    
    def trace_ray_real(self, h_start: float, lmbd: float, heights: List[float], 
                       aperture_radius: float = 10e-3) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Реальная трассировка луча через оптическую систему с учётом сферической аберрации
        
        Args:
            h_start: Начальная высота луча (расстояние от оптической оси)
            lmbd: Длина волны в метрах
            heights: Высоты микрорельефа для каждой линзы (в мкм)
            aperture_radius: Радиус апертуры линзы (по умолчанию 10 мм)
            
        Returns:
            (h_final, alpha_final, trace_history): Конечная высота, угол и история трассировки
        """
        h = h_start
        alpha = 0.0  # Луч идёт параллельно оптической оси
        trace_history = [(0, h, alpha)]  # (позиция_z, высота_h, угол_alpha)
        
        z_position = 0.0  # Начальная позиция по оси Z
        
        for i in range(1, self.count_linse + 1):
            # Расчёт параметров i-й линзы
            refractive_index = self.n_bk7(lmbd * 1e6)
            focus_0 = self.dataset['focus_0'][i] * 1e-3  # Переводим в метры
            harmonica = self.dataset['harmonica'][i]
            height = heights[i-1] * 1e-6  # Переводим в метры
            
            # Расчёт эффективной длины волны и оптической силы
            lambda_0 = height * (refractive_index - 1) / harmonica
            k = round((lambda_0 / lmbd) * harmonica)
            if k == 0:
                k = 0.5
                
            focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
            optic_power = 1 / focus
            
            # === РЕАЛЬНАЯ ТРАССИРОВКА С УЧЁТОМ СФЕРИЧЕСКОЙ АБЕРРАЦИИ ===
            
            # Для реальной линзы оптическая сила зависит от расстояния до оси
            # Сферическая аберрация: лучи на краю линзы преломляются сильнее
            
            # Нормализованная высота (от 0 до 1, где 1 - край линзы)
            normalized_h = abs(h) / aperture_radius if aperture_radius > 0 else 0
            
            # Коэффициент сферической аберрации (чем больше h, тем сильнее отклонение)
            # Используем разложение в ряд: D_real = D_paraxial * (1 + SA3 * h^2 + SA5 * h^4 + ...)
            # SA3 - коэффициент сферической аберрации 3-го порядка
            
            # Для гармонической линзы сферическая аберрация возникает из-за того,
            # что фазовый профиль не идеально сферический
            # Оценим сферическую аберрацию через отклонение от параксиального приближения
            
            # Коэффициент сферической аберрации (фиксированный, не зависит от апертуры)
            # Сферическая аберрация растёт с квадратом расстояния от оси
            # Для типичной линзы: SA3 ~ 0.1-1.0 мм^-2
            # Используем фиксированный коэффициент, чтобы аберрация зависела от реальной высоты луча
            SA3 = 5000.0  # м^-2 (фиксированный коэффициент)
            
            # Эффективная оптическая сила с учётом сферической аберрации
            # Аберрация растёт с квадратом расстояния от оси
            effective_power = optic_power * (1 + SA3 * h**2)
            
            # Преломление на линзе (с учётом сферической аберрации)
            h_new = h
            alpha_new = alpha - effective_power * h
            
            # Перенос до следующей линзы
            if i < self.count_linse:
                refractive_area = self.dataset['refractive_area'][f'{i}-{i+1}']
                dist = self.dataset['distance'][f'{i}-{i+1}'] * 1e-3  # Переводим в метры
                reduce_dist = dist / refractive_area
                
                z_position += reduce_dist
                h = h_new + alpha_new * reduce_dist
                alpha = alpha_new
            else:
                h = h_new
                alpha = alpha_new
            
            trace_history.append((z_position, h, alpha))
        
        return h, alpha, trace_history
    
    def trace_multiple_rays(self, lmbd: float, heights: List[float], 
                           num_rays: int = 20, 
                           aperture_radius: float = 10e-3) -> Dict:
        """
        Трассировка нескольких лучей на разных высотах
        
        Args:
            lmbd: Длина волны в метрах
            heights: Высоты микрорельефа для каждой линзы (в мкм)
            num_rays: Количество лучей для трассировки
            aperture_radius: Радиус апертуры линзы
            
        Returns:
            Словарь с результатами трассировки
        """
        # Распределяем лучи от оси до края апертуры
        ray_heights = np.linspace(0, aperture_radius, num_rays)
        
        results = {
            'ray_heights': [],
            'focus_positions': [],
            'transverse_aberration': [],
            'all_traces': []
        }
        
        for h_start in ray_heights:
            if h_start == 0:
                continue  # Пропускаем осевой луч
                
            h_final, alpha_final, trace_history = self.trace_ray_real(
                h_start, lmbd, heights, aperture_radius
            )
            
            # Находим положение фокуса (где луч пересекает оптическую ось)
            if abs(alpha_final) < 1e-10:
                focus_pos = float('inf')
            else:
                focus_pos = h_final / alpha_final
            
            # Поперечная аберрация (отклонение от параксиального фокуса)
            transverse_aberration = h_final  # На фиксированной плоскости
            
            results['ray_heights'].append(h_start)
            results['focus_positions'].append(focus_pos)
            results['transverse_aberration'].append(transverse_aberration)
            results['all_traces'].append(trace_history)
        
        results['ray_heights'] = np.array(results['ray_heights'])
        results['focus_positions'] = np.array(results['focus_positions'])
        results['transverse_aberration'] = np.array(results['transverse_aberration'])
        
        return results
    
    def find_focus_with_spherical_aberration(self, lmbd: float, heights: List[float],
                                            aperture_radius: float = 10e-3) -> Dict:
        """
        Нахождение положения фокуса с учётом сферической аберрации
        
        Returns:
            Словарь с различными характеристиками фокуса
        """
        # Трассируем луч на небольшой высоте (параксиальная область)
        h_paraxial = aperture_radius * 0.01  # 1% от апертуры
        h_final_p, alpha_final_p, _ = self.trace_ray_real(h_paraxial, lmbd, heights, aperture_radius)
        
        if abs(alpha_final_p) < 1e-10:
            paraxial_focus = float('inf')
        else:
            paraxial_focus = h_final_p / alpha_final_p
        
        # Трассируем луч на краю апертуры
        h_edge = aperture_radius * 0.99  # 99% от апертуры
        h_final_e, alpha_final_e, _ = self.trace_ray_real(h_edge, lmbd, heights, aperture_radius)
        
        if abs(alpha_final_e) < 1e-10:
            edge_focus = float('inf')
        else:
            edge_focus = h_final_e / alpha_final_e
        
        # Продольная сферическая аберрация
        longitudinal_SA = edge_focus - paraxial_focus
        
        # Трассируем несколько лучей для детального анализа
        multi_ray_results = self.trace_multiple_rays(lmbd, heights, 20, aperture_radius)
        
        # Находим положение наилучшей фокусировки (кружок наименьшего рассеяния)
        # Это середина между параксиальным и краевым фокусом
        best_focus = (paraxial_focus + edge_focus) / 2
        
        return {
            'paraxial_focus': paraxial_focus,
            'edge_focus': edge_focus,
            'longitudinal_SA': longitudinal_SA,
            'best_focus': best_focus,
            'multi_ray_results': multi_ray_results
        }


class ParaxialVsRealComparison:
    """Класс для сравнения параксиальной и реальной трассировки"""
    
    def __init__(self, dataset: Dict[DataSetKeys, any]):
        self.dataset = dataset
        self.count_linse = dataset['count_linse']
        self.real_tracer = RealRayTracer(dataset)
    
    def n_bk7(self, lambda_um: float) -> float:
        """Расчёт показателя преломления BK7 стекла"""
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
        l2 = lambda_um**2
        n2 = 1 + (B1 * l2) / (l2 - C1) + (B2 * l2) / (l2 - C2) + (B3 * l2) / (l2 - C3)
        return n2**0.5
    
    def trace_ray_paraxial(self, h_start: float, lmbd: float, heights: List[float]) -> Tuple[float, float]:
        """Параксиальная трассировка (существующая реализация)"""
        h = h_start
        alpha = 0.0
        
        for i in range(1, self.count_linse + 1):
            refractive_index = self.n_bk7(lmbd * 1e6)
            focus_0 = self.dataset['focus_0'][i] * 1e-3
            harmonica = self.dataset['harmonica'][i]
            height = heights[i-1] * 1e-6
            
            lambda_0 = height * (refractive_index - 1) / harmonica
            k = round((lambda_0 / lmbd) * harmonica)
            if k == 0:
                k = 0.5
                
            focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
            optic_power = 1 / focus
            
            h_new = h
            alpha_new = alpha - optic_power * h
            
            if i < self.count_linse:
                refractive_area = self.dataset['refractive_area'][f'{i}-{i+1}']
                dist = self.dataset['distance'][f'{i}-{i+1}'] * 1e-3
                reduce_dist = dist / refractive_area
                
                h = h_new + alpha_new * reduce_dist
                alpha = alpha_new
            else:
                h = h_new
                alpha = alpha_new
        
        return h, alpha
    
    def compare_tracing(self, lmbd: float, heights: List[float], 
                       aperture_radius: float = 10e-3,
                       num_rays: int = 20) -> Dict:
        """
        Сравнение параксиальной и реальной трассировки
        
        Args:
            lmbd: Длина волны в метрах
            heights: Высоты микрорельефа для каждой линзы (в мкм)
            aperture_radius: Радиус апертуры линзы
            num_rays: Количество лучей для трассировки
            
        Returns:
            Словарь с результатами сравнения
        """
        ray_heights = np.linspace(0.1e-3, aperture_radius, num_rays)  # От 0.1 мм до края
        
        comparison = {
            'ray_heights': [],
            'paraxial_focus': [],
            'real_focus': [],
            'longitudinal_aberration': [],
            'paraxial_alpha': [],
            'real_alpha': []
        }
        
        for h_start in ray_heights:
            # Параксиальная трассировка
            h_p, alpha_p = self.trace_ray_paraxial(h_start, lmbd, heights)
            if abs(alpha_p) > 1e-10:
                f_p = h_p / alpha_p
            else:
                f_p = float('inf')
            
            # Реальная трассировка
            h_r, alpha_r, _ = self.real_tracer.trace_ray_real(h_start, lmbd, heights, aperture_radius)
            if abs(alpha_r) > 1e-10:
                f_r = h_r / alpha_r
            else:
                f_r = float('inf')
            
            comparison['ray_heights'].append(h_start)
            comparison['paraxial_focus'].append(f_p)
            comparison['real_focus'].append(f_r)
            comparison['longitudinal_aberration'].append(f_r - f_p)
            comparison['paraxial_alpha'].append(alpha_p)
            comparison['real_alpha'].append(alpha_r)
        
        # Преобразуем в numpy массивы
        for key in comparison:
            comparison[key] = np.array(comparison[key])
        
        return comparison
    
    def compare_focus_vs_wavelength(self, heights: List[float],
                                   lambda_range: np.ndarray = None,
                                   aperture_radius: float = 10e-3) -> Dict:
        """
        Сравнение положения фокуса в зависимости от длины волны
        
        Args:
            heights: Высоты микрорельефа для каждой линзы (в мкм)
            lambda_range: Массив длин волн
            aperture_radius: Радиус апертуры
            
        Returns:
            Словарь с результатами сравнения
        """
        if lambda_range is None:
            lambda_range = np.linspace(
                self.dataset['lower_lambda'] * 1e-9,
                self.dataset['upper_lambda'] * 1e-9,
                50
            )
        
        comparison = {
            'wavelengths': [],
            'paraxial_focus': [],
            'real_focus_paraxial_region': [],
            'real_focus_edge': [],
            'longitudinal_SA': []
        }
        
        for lmbd in lambda_range:
            # Параксиальный фокус
            h_p, alpha_p = self.trace_ray_paraxial(1e-3, lmbd, heights)
            if abs(alpha_p) > 1e-10:
                f_p = abs(h_p / alpha_p)
            else:
                f_p = float('inf')
            
            # Реальный фокус с учётом сферической аберрации
            focus_info = self.real_tracer.find_focus_with_spherical_aberration(
                lmbd, heights, aperture_radius
            )
            
            comparison['wavelengths'].append(lmbd)
            comparison['paraxial_focus'].append(f_p)
            comparison['real_focus_paraxial_region'].append(focus_info['paraxial_focus'])
            comparison['real_focus_edge'].append(focus_info['edge_focus'])
            comparison['longitudinal_SA'].append(focus_info['longitudinal_SA'])
        
        # Преобразуем в numpy массивы
        for key in ['wavelengths', 'paraxial_focus', 'real_focus_paraxial_region', 
                    'real_focus_edge', 'longitudinal_SA']:
            comparison[key] = np.array(comparison[key])
        
        return comparison
    
    def plot_comparison(self, comparison: Dict, save_path: str = None):
        """
        Визуализация результатов сравнения
        
        Args:
            comparison: Словарь с результатами сравнения
            save_path: Путь для сохранения графика
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # График 1: Зависимость положения фокуса от высоты луча
        ax1 = axes[0, 0]
        ax1.plot(comparison['ray_heights'] * 1e3, comparison['paraxial_focus'] * 1e3, 
                'b-', linewidth=2, label='Параксиальная трассировка')
        ax1.plot(comparison['ray_heights'] * 1e3, comparison['real_focus'] * 1e3, 
                'r--', linewidth=2, label='Реальная трассировка')
        ax1.set_xlabel('Высота луча, мм')
        ax1.set_ylabel('Положение фокуса, мм')
        ax1.set_title('Зависимость фокуса от высоты луча')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: Продольная аберрация от высоты луча
        ax2 = axes[0, 1]
        ax2.plot(comparison['ray_heights'] * 1e3, comparison['longitudinal_aberration'] * 1e3, 
                'g-', linewidth=2)
        ax2.set_xlabel('Высота луча, мм')
        ax2.set_ylabel('Продольная аберрация, мм')
        ax2.set_title('Продольная сферическая аберрация')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # График 3: Угол отклонения от высоты луча
        ax3 = axes[1, 0]
        ax3.plot(comparison['ray_heights'] * 1e3, comparison['paraxial_alpha'], 
                'b-', linewidth=2, label='Параксиальный')
        ax3.plot(comparison['ray_heights'] * 1e3, comparison['real_alpha'], 
                'r--', linewidth=2, label='Реальный')
        ax3.set_xlabel('Высота луча, мм')
        ax3.set_ylabel('Угол отклонения, рад')
        ax3.set_title('Угол отклонения луча')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # График 4: Зависимость фокуса от длины волны (если есть данные)
        if 'wavelengths' in comparison:
            ax4 = axes[1, 1]
            ax4.plot(comparison['wavelengths'] * 1e9, comparison['paraxial_focus'] * 1e3, 
                    'b-', linewidth=2, label='Параксиальный')
            ax4.plot(comparison['wavelengths'] * 1e9, comparison['real_focus_paraxial_region'] * 1e3, 
                    'r--', linewidth=2, label='Реальный (паракс. область)')
            ax4.plot(comparison['wavelengths'] * 1e9, comparison['real_focus_edge'] * 1e3, 
                    'g-.', linewidth=2, label='Реальный (край)')
            ax4.set_xlabel('Длина волны, нм')
            ax4.set_ylabel('Положение фокуса, мм')
            ax4.set_title('Хроматическая зависимость фокуса')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранён в {save_path}")
        
        plt.show()


def run_full_comparison():
    """Запуск полного сравнения параксиальной и реальной трассировки"""
    print("="*60)
    print("СРАВНЕНИЕ ПАРАКСИАЛЬНОЙ И РЕАЛЬНОЙ ТРАССИРОВКИ")
    print("="*60)
    
    # Создаём датасет для 2-линзовой системы
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    # Оптимальные высоты (примерные)
    heights = [7.0, 7.5]  # в мкм
    
    # Создаём компаратор
    comparator = ParaxialVsRealComparison(dataset)
    
    # 1. Сравнение на фиксированной длине волны
    print("\n1. Сравнение на фиксированной длине волны (550 нм):")
    lmbd = 550e-9  # 550 нм
    comparison_height = comparator.compare_tracing(lmbd, heights, aperture_radius=10e-3)
    
    print(f"   Максимальная продольная аберрация: {np.max(np.abs(comparison_height['longitudinal_aberration']))*1e3:.4f} мм")
    
    # 2. Сравнение в зависимости от длины волны
    print("\n2. Сравнение в зависимости от длины волны:")
    comparison_lambda = comparator.compare_focus_vs_wavelength(heights, aperture_radius=10e-3)
    
    print(f"   Параксиальный фокальный отрезок: {(np.max(comparison_lambda['paraxial_focus']) - np.min(comparison_lambda['paraxial_focus']))*1e3:.4f} мм")
    print(f"   Реальный фокальный отрезок (паракс. область): {(np.max(comparison_lambda['real_focus_paraxial_region']) - np.min(comparison_lambda['real_focus_paraxial_region']))*1e3:.4f} мм")
    print(f"   Средняя продольная сферическая аберрация: {np.mean(np.abs(comparison_lambda['longitudinal_SA']))*1e3:.4f} мм")
    
    # 3. Визуализация
    print("\n3. Построение графиков...")
    
    # Объединяем результаты для визуализации
    full_comparison = {**comparison_height, **comparison_lambda}
    
    comparator.plot_comparison(full_comparison, save_path='results/two_linse/paraxial_vs_real_comparison.png')
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    return full_comparison


if __name__ == '__main__':
    run_full_comparison()
#!/usr/bin/env python3
"""
Быстрая демонстрация оптимизации и трассировки для 2х, 3х и 4х линзовых систем
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from optimizers.base import BaseLensOptimizer
from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from data.dataset_helper import DataSetHelper
from utils.unit import UnitType


class QuickRayTracer:
    """Упрощённый трассировщик лучей"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.count_linse = dataset['count_linse']
        
    def n_bk7(self, lambda_um: float) -> float:
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
        l2 = lambda_um**2
        n2 = 1 + (B1 * l2) / (l2 - C1) + (B2 * l2) / (l2 - C2) + (B3 * l2) / (l2 - C3)
        return n2**0.5
    
    def find_focus_position(self, lmbd: float, heights: List[float]) -> float:
        """Нахождение положения фокуса"""
        # Упрощённый расчёт для демонстрации
        h_test = 1e-3
        h = h_test
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
        
        if abs(alpha) < 1e-10:
            return float('inf')
        
        focus_pos = abs(h / alpha)
        return focus_pos


def quick_optimize_two_lens():
    """Быстрая оптимизация 2-линзовой системы"""
    print("Быстрая оптимизация 2-линзовой системы...")
    
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={f'1-2': 10}
    )
    
    optimizer = TwoLensOptimizer()
    
    # Быстрый перебор с крупным шагом
    h_range = np.linspace(6, 8, 10)
    min_foc_dist = float('inf')
    best_heights = [7.0, 7.0]
    
    for h1 in h_range:
        for h2 in h_range:
            try:
                lmbd_f_dict = optimizer.lmbd_focus_dict(
                    dataset=dataset, 
                    heights={UnitType.MICROMETER: [h1, h2]}, 
                    return_dict=True
                )
                foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                
                if foc_dist < min_foc_dist:
                    min_foc_dist = foc_dist
                    best_heights = [h1, h2]
                    
            except Exception:
                continue
    
    print(f"Лучшие параметры: h1={best_heights[0]:.2f} мкм, h2={best_heights[1]:.2f} мкм")
    print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")
    
    return dataset, best_heights, min_foc_dist


def quick_optimize_three_lens():
    """Быстрая оптимизация 3-линзовой системы"""
    print("Быстрая оптимизация 3-линзовой системы...")
    
    dataset = DataSetHelper.create_dataset(
        count_linse=3,
        focus_0={1: 300, 2: 300, 3: 300},
        harmonica={1: 7, 2: 7.5, 3: 8},
        distance={f'1-2': 10, f'2-3': 10}
    )
    
    optimizer = ThreeLensOptimizer()
    
    # Быстрый перебор
    h_range = np.linspace(6, 8, 5)
    min_foc_dist = float('inf')
    best_heights = [7.0, 7.0, 7.0]
    
    for h1 in h_range:
        for h2 in h_range:
            for h3 in h_range:
                try:
                    lmbd_f_dict = optimizer.lmbd_focus_dict(
                        dataset=dataset, 
                        heights={UnitType.MICROMETER: [h1, h2, h3]}, 
                        return_dict=True
                    )
                    foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                    
                    if foc_dist < min_foc_dist:
                        min_foc_dist = foc_dist
                        best_heights = [h1, h2, h3]
                        
                except Exception:
                    continue
    
    print(f"Лучшие параметры: h1={best_heights[0]:.2f} мкм, h2={best_heights[1]:.2f} мкм, h3={best_heights[2]:.2f} мкм")
    print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")
    
    return dataset, best_heights, min_foc_dist


def quick_trace_rays(dataset, heights, system_name):
    """Быстрая трассировка лучей"""
    tracer = QuickRayTracer(dataset)
    
    lambda_range = np.linspace(dataset['lower_lambda'] * 1e-9, 
                             dataset['upper_lambda'] * 1e-9, 20)
    
    focus_positions = []
    
    print(f"Трассировка лучей для {system_name}...")
    
    for lmbd in lambda_range:
        focus_pos = tracer.find_focus_position(lmbd, heights)
        focus_positions.append(focus_pos)
    
    focus_positions = np.array(focus_positions)
    focus_width = np.max(focus_positions) - np.min(focus_positions)
    
    results = {
        'lambda_range': lambda_range * 1e9,
        'focus_positions': focus_positions * 1000,
        'focus_width': focus_width * 1000,
        'heights': heights
    }
    
    print(f"Ширина фокального отрезка методом трассировки: {focus_width*1000:.4f} мм")
    
    return results


def plot_quick_results(count_linse, dataset, best_heights, min_foc_dist, tracing_results):
    """Построение быстрых графиков"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Параксиальный расчёт
    if count_linse == 2:
        optimizer = TwoLensOptimizer()
    else:
        optimizer = ThreeLensOptimizer()
    
    lmbd_f_dict = optimizer.lmbd_focus_dict(
        dataset=dataset,
        heights={UnitType.MICROMETER: best_heights},
        return_dict=True
    )
    
    lmbd_list = np.array(list(lmbd_f_dict.keys())) * 1e9
    f_list = np.array(list(lmbd_f_dict.values())) * 1e3
    
    ax1.plot(lmbd_list, f_list, 'b-', linewidth=2, label='Параксиальный расчёт')
    ax1.set_xlabel('Длина волны, нм')
    ax1.set_ylabel('Фокусное расстояние, мм')
    ax1.set_title(f'Параксиальный расчёт ({count_linse} линзы)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Трассировка
    ax2.plot(tracing_results['lambda_range'], tracing_results['focus_positions'], 'r-', linewidth=2, label='Трассировка лучей')
    ax2.set_xlabel('Длина волны, нм')
    ax2.set_ylabel('Положение фокуса, мм')
    ax2.set_title(f'Трассировка лучей ({count_linse} линзы)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    textstr = f'Оптимальные высоты: {[f"{h:.2f}" for h in best_heights]} мкм\n' \
             f'Фок. отрезок (паракс.): {min_foc_dist*1000:.4f} мм\n' \
             f'Фок. отрезок (трасс.): {tracing_results["focus_width"]:.4f} мм'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'results/quick_analysis_{count_linse}_lenses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Графики сохранены в results/quick_analysis_{count_linse}_lenses.png")


def main():
    """Основная функция быстрой демонстрации"""
    print("Быстрая демонстрация оптимизации и трассировки...")
    
    # 2-линзовая система
    print("\n" + "="*50)
    print("2-ЛИНЗОВАЯ СИСТЕМА")
    print("="*50)
    
    dataset_2, heights_2, foc_dist_2 = quick_optimize_two_lens()
    tracing_2 = quick_trace_rays(dataset_2, heights_2, "2-линзовая система")
    plot_quick_results(2, dataset_2, heights_2, foc_dist_2, tracing_2)
    
    # 3-линзовая система
    print("\n" + "="*50)
    print("3-ЛИНЗОВАЯ СИСТЕМА")
    print("="*50)
    
    dataset_3, heights_3, foc_dist_3 = quick_optimize_three_lens()
    tracing_3 = quick_trace_rays(dataset_3, heights_3, "3-линзовая система")
    plot_quick_results(3, dataset_3, heights_3, foc_dist_3, tracing_3)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЫСТРОЙ ДЕМОНСТРАЦИИ")
    print("="*60)
    
    print(f"\n2-линзовая система:")
    print(f"  Оптимальные высоты: {[f'{h:.2f} мкм' for h in heights_2]}")
    print(f"  Фокальный отрезок (параксиальный): {foc_dist_2*1000:.4f} мм")
    print(f"  Фокальный отрезок (трассировка): {tracing_2['focus_width']:.4f} мм")
    
    print(f"\n3-линзовая система:")
    print(f"  Оптимальные высоты: {[f'{h:.2f} мкм' for h in heights_3]}")
    print(f"  Фокальный отрезок (параксиальный): {foc_dist_3*1000:.4f} мм")
    print(f"  Фокальный отрезок (трассировка): {tracing_3['focus_width']:.4f} мм")


if __name__ == "__main__":
    main()
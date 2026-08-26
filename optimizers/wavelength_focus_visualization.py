"""
Визуализация зависимости фокусного расстояния от длины волны
для 2-х, 3-х, 4-х и 5-ти линзовых систем (аналогично focus_analysis_2_lenses.png)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from optimizers.four_lens import FourLensOptimizer
from optimizers.five_lens import FiveLensOptimizer
from optimizers.real_tracing import RealRayTracer


class WavelengthFocusVisualizer:
    """Визуализация зависимости фокусного расстояния от длины волны"""
    
    def __init__(self):
        self.optimizers = {
            2: TwoLensOptimizer(),
            3: ThreeLensOptimizer(),
            4: FourLensOptimizer(),
            5: FiveLensOptimizer()
        }
        
        self.datasets = {}
        self.results = {}
        
    def create_datasets(self):
        """Создание датасетов для всех систем"""
        # 2-линзовая система
        self.datasets[2] = DataSetHelper.create_dataset(
            count_linse=2,
            focus_0={1: 200, 2: 200},
            harmonica={1: 7, 2: 7.5},
            distance={f'1-2': 10}
        )
        
        # 3-линзовая система
        self.datasets[3] = DataSetHelper.create_dataset(
            count_linse=3,
            focus_0={1: 300, 2: 300, 3: 300},
            harmonica={1: 7, 2: 7, 3: 7},
            distance={f'1-2': 10, f'2-3': 10}
        )
        
        # 4-линзовая система
        self.datasets[4] = DataSetHelper.create_dataset(
            count_linse=4,
            focus_0={1: 400, 2: 400, 3: 400, 4: 400},
            harmonica={1: 7, 2: 7, 3: 7, 4: 7},
            distance={f'1-2': 10, f'2-3': 10, f'3-4': 10}
        )
        
        # 5-линзовая система
        self.datasets[5] = DataSetHelper.create_dataset(
            count_linse=5,
            focus_0={1: 500, 2: 500, 3: 500, 4: 500, 5: 500},
            harmonica={1: 7, 2: 7, 3: 7, 4: 7, 5: 7},
            distance={f'1-2': 10, f'2-3': 10, f'3-4': 10, f'4-5': 10}
        )
        
    def calculate_focus_vs_wavelength(self, count_linse: int, 
                                     heights: List[float] = None) -> Dict:
        """Расчёт фокусного расстояния в зависимости от длины волны"""
        optimizer = self.optimizers[count_linse]
        dataset = self.datasets[count_linse]
        
        if heights is None:
            # Используем стандартные высоты
            heights = [7.0] * count_linse
        
        # Получаем данные (параксиальный расчёт)
        lmbd_f_dict = optimizer.lmbd_focus_dict(
            dataset=dataset,
            heights={UnitType.MICROMETER: heights},
            return_dict=True
        )
        
        # Проверяем формат словаря
        if isinstance(lmbd_f_dict, dict) and 'lambda' in lmbd_f_dict:
            # Формат: {'lambda': [...], 'focus': [...]}
            lambda_values = np.array(lmbd_f_dict['lambda']) * 1e9  # В нм
            focus_values = np.array(lmbd_f_dict['focus']) * 1e3   # В мм
        else:
            # Формат: {длина_волны: фокус}
            lambda_values = np.array([float(k) for k in lmbd_f_dict.keys()]) * 1e9  # В нм
            focus_values = np.array([float(v) for v in lmbd_f_dict.values()]) * 1e3   # В мм
        
        # Берём модуль фокусного расстояния (убираем знак)
        focus_values = np.abs(focus_values)
        
        # Рассчитываем хроматическую аберрацию
        chromatic_aberration = (max(focus_values) - min(focus_values))
        
        # Рассчитываем сферическую аберрацию с помощью реальной трассировки
        real_tracer = RealRayTracer(dataset)
        real_focus_values = []
        spherical_aberrations = []
        
        for lmbd in lambda_values * 1e-9:
            focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
            # Берём модуль фокусного расстояния
            real_focus_values.append(abs(focus_info['paraxial_focus']) * 1e3)
            spherical_aberrations.append(abs(focus_info['longitudinal_SA']) * 1e3)
        
        real_focus_values = np.array(real_focus_values)
        spherical_aberrations = np.array(spherical_aberrations)
        
        # Рассчитываем фокальный отрезок для реальной трассировки
        real_focus_width = max(real_focus_values) - min(real_focus_values)
        
        return {
            'lambda': lambda_values,
            'paraxial_focus': focus_values,
            'real_focus': real_focus_values,
            'spherical_aberration': spherical_aberrations,
            'chromatic_aberration': chromatic_aberration,
            'real_focus_width': real_focus_width,
            'avg_spherical_aberration': np.mean(spherical_aberrations),
            'heights': heights
        }
    
    def plot_wavelength_vs_focus(self):
        """Построение графиков зависимости фокусного расстояния от длины волны
        (аналогично focus_analysis_2_lenses.png)"""
        print("Расчёт зависимости фокусного расстояния от длины волны...")
        
        systems = [2, 3, 4, 5]
        save_paths = {
            2: 'two_linse',
            3: 'three_linse',
            4: 'four_linse',
            5: 'five_linse'
        }
        
        for count_linse in systems:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Рассчитываем данные
            result = self.calculate_focus_vs_wavelength(count_linse)
            self.results[count_linse] = result
            
            # График 1: Параксиальный расчёт
            ax1.plot(result['lambda'], result['paraxial_focus'], 
                    'b-', linewidth=2, label='Параксиальный расчёт')
            ax1.set_xlabel('Длина волны, нм')
            ax1.set_ylabel('Фокусное расстояние, мм')
            ax1.set_title(f'Параксиальный расчёт ({count_linse} линзы)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # График 2: Реальная трассировка
            ax2.plot(result['lambda'], result['real_focus'], 
                    'r-', linewidth=2, label='Реальная трассировка')
            ax2.set_xlabel('Длина волны, нм')
            ax2.set_ylabel('Фокусное расстояние, мм')
            ax2.set_title(f'Реальная трассировка ({count_linse} линзы)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Добавление информации о результатах
            textstr = (f'Оптимальные высоты: {[f"{h:.2f}" for h in result["heights"]]} мкм\n'
                      f'Фок. отрезок (паракс.): {result["chromatic_aberration"]:.4f} мм\n'
                      f'Фок. отрезок (реальный): {result["real_focus_width"]:.4f} мм\n'
                      f'Сфер. аберрация: {result["avg_spherical_aberration"]:.2f} мм')
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Сохраняем график
            os.makedirs(f'results/{save_paths[count_linse]}', exist_ok=True)
            plt.savefig(f'results/{save_paths[count_linse]}/focus_analysis_{count_linse}_lenses.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"График сохранён в results/{save_paths[count_linse]}/focus_analysis_{count_linse}_lenses.png")
    
    def plot_aberration_comparison(self):
        """Сравнение аберраций для всех систем"""
        print("Построение сравнения аберраций...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        systems = [2, 3, 4, 5]
        system_names = ['2-линзовая', '3-линзовая', '4-линзовая', '5-линзовая']
        
        chromatic_aberrations = []
        spherical_aberrations = []
        real_focus_widths = []
        
        for count_linse in systems:
            result = self.results[count_linse]
            chromatic_aberrations.append(result['chromatic_aberration'])
            spherical_aberrations.append(result['avg_spherical_aberration'])
            real_focus_widths.append(result['real_focus_width'])
        
        x = np.arange(len(systems))
        width = 0.25
        
        # График аберраций
        ax1.bar(x - width, chromatic_aberrations, width, label='Хроматическая', 
               color='blue', alpha=0.8)
        ax1.bar(x, spherical_aberrations, width, label='Сферическая', 
               color='red', alpha=0.8)
        ax1.bar(x + width, real_focus_widths, width, label='Фок. отрезок (реальный)', 
               color='green', alpha=0.8)
        
        ax1.set_xlabel('Тип системы')
        ax1.set_ylabel('Аберрация, мм')
        ax1.set_title('Сравнение аберраций')
        ax1.set_xticks(x)
        ax1.set_xticklabels(system_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, (ca, sa, rw) in enumerate(zip(chromatic_aberrations, spherical_aberrations, real_focus_widths)):
            ax1.text(i - width, ca + 0.1, f'{ca:.2f}', ha='center', va='bottom', fontsize=7)
            ax1.text(i, sa + 0.1, f'{sa:.2f}', ha='center', va='bottom', fontsize=7)
            ax1.text(i + width, rw + 0.1, f'{rw:.2f}', ha='center', va='bottom', fontsize=7)
        
        # График отношения аберраций
        ratios = [sa / ca if ca > 0 else 0 for ca, sa in zip(chromatic_aberrations, spherical_aberrations)]
        
        ax2.bar(system_names, ratios, color='green', alpha=0.8)
        ax2.set_xlabel('Тип системы')
        ax2.set_ylabel('Отношение сфер./хром. аберрация')
        ax2.set_title('Отношение аберраций')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, ratio in enumerate(ratios):
            ax2.text(i, ratio + 0.01, f'{ratio:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/comparison/aberration_comparison_all_systems.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("График сохранён в results/comparison/aberration_comparison_all_systems.png")
    
    def plot_heights_comparison(self):
        """Сравнение оптимальных высот для всех систем"""
        print("Построение сравнения высот...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = [2, 3, 4, 5]
        system_names = ['2-линзовая', '3-линзовая', '4-линзовая', '5-линзовая']
        
        for count_linse in systems:
            result = self.results[count_linse]
            heights = result['heights']
            
            lens_numbers = np.arange(1, count_linse + 1)
            ax.plot(lens_numbers, heights, 'o-', linewidth=2, 
                   label=f'{count_linse}-линзовая', markersize=8)
        
        ax.set_xlabel('Номер линзы')
        ax.set_ylabel('Высота микрорельефа, мкм')
        ax.set_title('Оптимальные высоты микрорельефа')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(1, 6))
        
        plt.tight_layout()
        plt.savefig('results/comparison/heights_comparison_all_systems.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("График сохранён в results/comparison/heights_comparison_all_systems.png")
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("="*80)
        print("АНАЛИЗ ЗАВИСИМОСТИ ФОКУСНОГО РАССТОЯНИЯ ОТ ДЛИНЫ ВОЛНЫ")
        print("="*80)
        
        # Создаём датасеты
        self.create_datasets()
        
        # Строим графики (аналогично focus_analysis_2_lenses.png)
        self.plot_wavelength_vs_focus()
        self.plot_aberration_comparison()
        self.plot_heights_comparison()
        
        # Выводим итоговую таблицу
        print("\n" + "="*80)
        print("ИТОГОВАЯ ТАБЛИЦА")
        print("="*80)
        print(f"{'Система':<15} {'Хром. аберр., мм':<20} {'Сфер. аберр., мм':<20} {'Фок. отрезок (реальный), мм':<25}")
        print("-"*80)
        
        for count_linse in [2, 3, 4, 5]:
            result = self.results[count_linse]
            print(f"{count_linse}-линзовая    "
                  f"{result['chromatic_aberration']:<20.4f} "
                  f"{result['avg_spherical_aberration']:<20.4f} "
                  f"{result['real_focus_width']:<25.4f}")


def main():
    """Основная функция"""
    visualizer = WavelengthFocusVisualizer()
    visualizer.run_full_analysis()


if __name__ == '__main__':
    main()
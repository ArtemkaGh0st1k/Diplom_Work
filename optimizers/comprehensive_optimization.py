"""
Комплексная оптимизация для 2-х, 3-х, 4-х и 5-ти линзовых систем
Сравнение параксиальной и реальной оптимизации
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from optimizers.four_lens import FourLensOptimizer
from optimizers.five_lens import FiveLensOptimizer
from optimizers.real_tracing import RealRayTracer


class ComprehensiveOptimizer:
    """Комплексная оптимизация для всех систем"""
    
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
    
    def optimize_paraxial(self, count_linse: int, h_range: np.ndarray = None) -> Tuple[List[float], float]:
        """Параксиальная оптимизация"""
        optimizer = self.optimizers[count_linse]
        dataset = self.datasets[count_linse]
        
        if h_range is None:
            h_range = np.linspace(5.0, 10.0, 50)
        
        min_foc_dist = float('inf')
        best_heights = [7.0] * count_linse
        
        print(f"Параксиальная оптимизация {count_linse}-линзовой системы...")
        start_time = time.time()
        
        if count_linse == 2:
            for h2 in h_range:
                try:
                    lmbd_f_dict = optimizer.lmbd_focus_dict(
                        dataset=dataset, 
                        heights={UnitType.MICROMETER: [7.0, h2]}, 
                        return_dict=True
                    )
                    foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                    
                    if foc_dist < min_foc_dist:
                        min_foc_dist = foc_dist
                        best_heights = [7.0, h2]
                        
                except Exception:
                    continue
        
        elif count_linse == 3:
            for h2 in h_range[::2]:  # Шаг 2 для ускорения
                for h3 in h_range[::2]:
                    try:
                        lmbd_f_dict = optimizer.lmbd_focus_dict(
                            dataset=dataset, 
                            heights={UnitType.MICROMETER: [7.0, h2, h3]}, 
                            return_dict=True
                        )
                        foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                        
                        if foc_dist < min_foc_dist:
                            min_foc_dist = foc_dist
                            best_heights = [7.0, h2, h3]
                            
                    except Exception:
                        continue
        
        elif count_linse == 4:
            for h2 in h_range[::3]:
                for h3 in h_range[::3]:
                    for h4 in h_range[::3]:
                        try:
                            lmbd_f_dict = optimizer.lmbd_focus_dict(
                                dataset=dataset, 
                                heights={UnitType.MICROMETER: [7.0, h2, h3, h4]}, 
                                return_dict=True
                            )
                            foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                            
                            if foc_dist < min_foc_dist:
                                min_foc_dist = foc_dist
                                best_heights = [7.0, h2, h3, h4]
                                
                        except Exception:
                            continue
        
        elif count_linse == 5:
            # Для 5-линзовой системы используем более грубый перебор
            h_range_coarse = np.linspace(5.0, 10.0, 10)
            for h2 in h_range_coarse:
                for h3 in h_range_coarse:
                    for h4 in h_range_coarse:
                        for h5 in h_range_coarse:
                            try:
                                lmbd_f_dict = optimizer.lmbd_focus_dict(
                                    dataset=dataset, 
                                    heights={UnitType.MICROMETER: [7.0, h2, h3, h4, h5]}, 
                                    return_dict=True
                                )
                                foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                                
                                if foc_dist < min_foc_dist:
                                    min_foc_dist = foc_dist
                                    best_heights = [7.0, h2, h3, h4, h5]
                                    
                            except Exception:
                                continue
        
        end_time = time.time()
        print(f"Параксиальная оптимизация {count_linse}-линз завершена за {(end_time - start_time):.2f} сек")
        print(f"Лучшие параметры: {[f'{h:.2f}' for h in best_heights]} мкм")
        print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")
        
        return best_heights, min_foc_dist
    
    def optimize_real_tracing(self, count_linse: int, h_range: np.ndarray = None) -> Tuple[List[float], float]:
        """Оптимизация по реальной трассировке"""
        dataset = self.datasets[count_linse]
        real_tracer = RealRayTracer(dataset)
        
        if h_range is None:
            h_range = np.linspace(5.0, 10.0, 30)
        
        min_focus_width = float('inf')
        best_heights = [7.0] * count_linse
        
        print(f"Реальная оптимизация {count_linse}-линзовой системы...")
        start_time = time.time()
        
        # Диапазон длин волн для трассировки
        lambda_range = np.linspace(dataset['lower_lambda'] * 1e-9, 
                                 dataset['upper_lambda'] * 1e-9, 20)
        
        if count_linse == 2:
            for h2 in h_range:
                try:
                    focus_positions = []
                    for lmbd in lambda_range:
                        focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, [7.0, h2])
                        focus_positions.append(abs(focus_info['paraxial_focus']))
                    
                    focus_width = max(focus_positions) - min(focus_positions)
                    
                    if focus_width < min_focus_width:
                        min_focus_width = focus_width
                        best_heights = [7.0, h2]
                        
                except Exception:
                    continue
        
        elif count_linse == 3:
            for h2 in h_range[::2]:
                for h3 in h_range[::2]:
                    try:
                        focus_positions = []
                        for lmbd in lambda_range:
                            focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, [7.0, h2, h3])
                            focus_positions.append(abs(focus_info['paraxial_focus']))
                        
                        focus_width = max(focus_positions) - min(focus_positions)
                        
                        if focus_width < min_focus_width:
                            min_focus_width = focus_width
                            best_heights = [7.0, h2, h3]
                            
                    except Exception:
                        continue
        
        elif count_linse == 4:
            for h2 in h_range[::2]:
                for h3 in h_range[::2]:
                    for h4 in h_range[::2]:
                        try:
                            focus_positions = []
                            for lmbd in lambda_range:
                                focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, [7.0, h2, h3, h4])
                                focus_positions.append(abs(focus_info['paraxial_focus']))
                            
                            focus_width = max(focus_positions) - min(focus_positions)
                            
                            if focus_width < min_focus_width:
                                min_focus_width = focus_width
                                best_heights = [7.0, h2, h3, h4]
                                
                        except Exception:
                            continue
        
        elif count_linse == 5:
            # Для 5-линзовой системы используем более грубый перебор
            h_range_coarse = np.linspace(5.0, 10.0, 8)
            for h2 in h_range_coarse:
                for h3 in h_range_coarse:
                    for h4 in h_range_coarse:
                        for h5 in h_range_coarse:
                            try:
                                focus_positions = []
                                for lmbd in lambda_range:
                                    focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, [7.0, h2, h3, h4, h5])
                                    focus_positions.append(abs(focus_info['paraxial_focus']))
                                
                                focus_width = max(focus_positions) - min(focus_positions)
                                
                                if focus_width < min_focus_width:
                                    min_focus_width = focus_width
                                    best_heights = [7.0, h2, h3, h4, h5]
                                    
                            except Exception:
                                continue
        
        end_time = time.time()
        print(f"Реальная оптимизация {count_linse}-линз завершена за {(end_time - start_time):.2f} сек")
        print(f"Лучшие параметры: {[f'{h:.2f}' for h in best_heights]} мкм")
        print(f"Минимальный фокальный отрезок: {min_focus_width*1000:.4f} мм")
        
        return best_heights, min_focus_width
    
    def calculate_wavelength_dependence(self, count_linse: int, heights: List[float]) -> Dict:
        """Расчёт зависимости фокусного расстояния от длины волны"""
        optimizer = self.optimizers[count_linse]
        dataset = self.datasets[count_linse]
        real_tracer = RealRayTracer(dataset)
        
        # Параксиальный расчёт
        lmbd_f_dict = optimizer.lmbd_focus_dict(
            dataset=dataset,
            heights={UnitType.MICROMETER: heights},
            return_dict=True
        )
        
        # Проверяем формат словаря
        if isinstance(lmbd_f_dict, dict) and 'lambda' in lmbd_f_dict:
            lambda_values = np.array(lmbd_f_dict['lambda']) * 1e9  # В нм
            paraxial_focus = np.array(lmbd_f_dict['focus']) * 1e3   # В мм
        else:
            lambda_values = np.array([float(k) for k in lmbd_f_dict.keys()]) * 1e9  # В нм
            paraxial_focus = np.array([float(v) for v in lmbd_f_dict.values()]) * 1e3   # В мм
        
        # Берём модуль фокусного расстояния
        paraxial_focus = np.abs(paraxial_focus)
        
        # Реальная трассировка
        real_focus = []
        spherical_aberrations = []
        
        for lmbd in lambda_values * 1e-9:
            focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
            real_focus.append(abs(focus_info['paraxial_focus']) * 1e3)
            spherical_aberrations.append(abs(focus_info['longitudinal_SA']) * 1e3)
        
        real_focus = np.array(real_focus)
        spherical_aberrations = np.array(spherical_aberrations)
        
        # Рассчитываем фокальные отрезки
        paraxial_width = max(paraxial_focus) - min(paraxial_focus)
        real_width = max(real_focus) - min(real_focus)
        
        return {
            'lambda': lambda_values,
            'paraxial_focus': paraxial_focus,
            'real_focus': real_focus,
            'spherical_aberration': spherical_aberrations,
            'paraxial_width': paraxial_width,
            'real_width': real_width,
            'heights': heights
        }
    
    def run_optimization(self):
        """Запуск комплексной оптимизации"""
        print("="*80)
        print("КОМПЛЕКСНАЯ ОПТИМИЗАЦИЯ ДЛЯ 2-Х, 3-Х, 4-Х И 5-ТИ ЛИНЗОВЫХ СИСТЕМ")
        print("="*80)
        
        # Создаём датасеты
        self.create_datasets()
        
        systems = [2, 3, 4, 5]
        
        for count_linse in systems:
            print(f"\n{'='*60}")
            print(f"ОПТИМИЗАЦИЯ {count_linse}-ЛИНЗОВОЙ СИСТЕМЫ")
            print(f"{'='*60}")
            
            # Параксиальная оптимизация
            paraxial_heights, paraxial_width = self.optimize_paraxial(count_linse)
            
            # Реальная оптимизация
            real_heights, real_width = self.optimize_real_tracing(count_linse)
            
            # Расчёт зависимостей
            paraxial_data = self.calculate_wavelength_dependence(count_linse, paraxial_heights)
            real_data = self.calculate_wavelength_dependence(count_linse, real_heights)
            
            # Сохранение результатов
            self.results[count_linse] = {
                'paraxial': {
                    'heights': paraxial_heights,
                    'width': paraxial_width,
                    'data': paraxial_data
                },
                'real': {
                    'heights': real_heights,
                    'width': real_width,
                    'data': real_data
                }
            }
            
            # Визуализация
            self.plot_optimization_results(count_linse)
    
    def plot_optimization_results(self, count_linse: int):
        """Построение результатов оптимизации"""
        result = self.results[count_linse]
        
        save_paths = {
            2: 'two_linse',
            3: 'three_linse',
            4: 'four_linse',
            5: 'five_linse'
        }
        
        # Создаём директорию
        os.makedirs(f'results/{save_paths[count_linse]}', exist_ok=True)
        
        # График 1: Сравнение высот
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Высоты
        lens_numbers = np.arange(1, count_linse + 1)
        ax1.plot(lens_numbers, result['paraxial']['heights'], 'bo-', linewidth=2, 
                label=f'Параксиальная оптимизация', markersize=8)
        ax1.plot(lens_numbers, result['real']['heights'], 'ro-', linewidth=2, 
                label=f'Реальная оптимизация', markersize=8)
        ax1.set_xlabel('Номер линзы')
        ax1.set_ylabel('Высота микрорельефа, мкм')
        ax1.set_title(f'Оптимальные высоты ({count_linse} линзы)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(lens_numbers)
        
        # Добавляем значения
        for i, (h_par, h_real) in enumerate(zip(result['paraxial']['heights'], result['real']['heights'])):
            ax1.text(i+1, h_par + 0.1, f'{h_par:.2f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i+1, h_real + 0.1, f'{h_real:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Зависимость фокусного расстояния от длины волны (параксиальная оптимизация)
        data = result['paraxial']['data']
        ax2.plot(data['lambda'], data['paraxial_focus'], 'b-', linewidth=2, 
                label='Параксиальный расчёт')
        ax2.plot(data['lambda'], data['real_focus'], 'r-', linewidth=2, 
                label='Реальная трассировка')
        ax2.set_xlabel('Длина волны, нм')
        ax2.set_ylabel('Фокусное расстояние, мм')
        ax2.set_title(f'Зависимость f(λ) - Параксиальная оптимизация')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Добавляем информацию
        textstr = (f'Высоты: {[f"{h:.2f}" for h in result["paraxial"]["heights"]]} мкм\n'
                  f'Фок. отрезок (паракс.): {result["paraxial"]["width"]*1000:.4f} мм\n'
                  f'Фок. отрезок (реальный): {data["real_width"]:.4f} мм')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Зависимость фокусного расстояния от длины волны (реальная оптимизация)
        data = result['real']['data']
        ax3.plot(data['lambda'], data['paraxial_focus'], 'b-', linewidth=2, 
                label='Параксиальный расчёт')
        ax3.plot(data['lambda'], data['real_focus'], 'r-', linewidth=2, 
                label='Реальная трассировка')
        ax3.set_xlabel('Длина волны, нм')
        ax3.set_ylabel('Фокусное расстояние, мм')
        ax3.set_title(f'Зависимость f(λ) - Реальная оптимизация')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Добавляем информацию
        textstr = (f'Высоты: {[f"{h:.2f}" for h in result["real"]["heights"]]} мкм\n'
                  f'Фок. отрезок (реальный): {result["real"]["width"]*1000:.4f} мм\n'
                  f'Фок. отрезок (паракс.): {data["paraxial_width"]:.4f} мм')
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Сравнение фокальных отрезков
        categories = ['Параксиальная\nоптимизация', 'Реальная\nоптимизация']
        paraxial_widths = [result['paraxial']['width']*1000, result['real']['data']['paraxial_width']]
        real_widths = [result['paraxial']['data']['real_width'], result['real']['width']*1000]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, paraxial_widths, width, label='Параксиальный расчёт', 
               color='blue', alpha=0.7)
        ax4.bar(x + width/2, real_widths, width, label='Реальная трассировка', 
               color='red', alpha=0.7)
        
        ax4.set_xlabel('Тип оптимизации')
        ax4.set_ylabel('Фокальный отрезок, мм')
        ax4.set_title(f'Сравнение фокальных отрезков ({count_linse} линзы)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Добавляем значения
        for i, (pw, rw) in enumerate(zip(paraxial_widths, real_widths)):
            ax4.text(i - width/2, pw + 0.1, f'{pw:.2f}', ha='center', va='bottom', fontsize=8)
            ax4.text(i + width/2, rw + 0.1, f'{rw:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_paths[count_linse]}/comprehensive_optimization_{count_linse}_lenses.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"График сохранён в results/{save_paths[count_linse]}/comprehensive_optimization_{count_linse}_lenses.png")
    
    def print_summary(self):
        """Вывод итоговой таблицы"""
        print("\n" + "="*100)
        print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        print("="*100)
        print(f"{'Система':<12} {'Тип':<15} {'Высоты, мкм':<25} {'Фок. отрезок, мм':<20}")
        print("-"*100)
        
        for count_linse in [2, 3, 4, 5]:
            result = self.results[count_linse]
            
            # Параксиальная оптимизация
            heights_par = result['paraxial']['heights']
            width_par = result['paraxial']['width'] * 1000
            print(f"{count_linse}-линзовая  Параксиальная    {[f'{h:.2f}' for h in heights_par]}{' ' * (25 - len(str([f'{h:.2f}' for h in heights_par])))} {width_par:<20.4f}")
            
            # Реальная оптимизация
            heights_real = result['real']['heights']
            width_real = result['real']['width'] * 1000
            print(f"{count_linse}-линзовая  Реальная         {[f'{h:.2f}' for h in heights_real]}{' ' * (25 - len(str([f'{h:.2f}' for h in heights_real])))} {width_real:<20.4f}")
            
            # Разница в высотах
            height_diff = np.array(heights_real) - np.array(heights_par)
            print(f"{' ' * 12} Разница высот     {[f'{h:+.2f}' for h in height_diff]}{' ' * (25 - len(str([f'{h:+.2f}' for h in height_diff])))}")
            print()
        
        print("="*100)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print("="*100)
        
        for count_linse in [2, 3, 4, 5]:
            result = self.results[count_linse]
            
            paraxial_width = result['paraxial']['width'] * 1000
            real_width_par = result['paraxial']['data']['real_width']
            real_width = result['real']['width'] * 1000
            paraxial_width_real = result['real']['data']['paraxial_width']
            
            improvement_paraxial = ((real_width_par - paraxial_width) / real_width_par) * 100
            improvement_real = ((paraxial_width_real - real_width) / paraxial_width_real) * 100
            
            print(f"\n{count_linse}-линзовая система:")
            print(f"  Параксиальная оптимизация:")
            print(f"    - Фок. отрезок (паракс.): {paraxial_width:.4f} мм")
            print(f"    - Фок. отрезок (реальный): {real_width_par:.4f} мм")
            print(f"    - Ошибка: {real_width_par - paraxial_width:.4f} мм ({improvement_paraxial:+.2f}%)")
            
            print(f"  Реальная оптимизация:")
            print(f"    - Фок. отрезок (реальный): {real_width:.4f} мм")
            print(f"    - Фок. отрезок (паракс.): {paraxial_width_real:.4f} мм")
            print(f"    - Улучшение: {paraxial_width_real - real_width:.4f} мм ({improvement_real:+.2f}%)")
            
            # Средние высоты
            avg_height_par = np.mean(result['paraxial']['heights'])
            avg_height_real = np.mean(result['real']['heights'])
            print(f"  Средние высоты: {avg_height_par:.2f} мкм (паракс.) → {avg_height_real:.2f} мкм (реальная)")


def main():
    """Основная функция"""
    optimizer = ComprehensiveOptimizer()
    optimizer.run_optimization()
    optimizer.print_summary()


if __name__ == '__main__':
    main()
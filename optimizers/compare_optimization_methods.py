"""
Модуль для сравнения оптимизации с параксиальной и реальной трассировкой.
Сравниваем найденные оптимальные высоты и полученные фокальные отрезки.
"""

import sys
import os

from optimizers.five_lens import FiveLensOptimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from utils.keys import DataSetKeys

# Импортируем существующие оптимизаторы
from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from optimizers.four_lens import FourLensOptimizer
from optimizers.real_tracing import RealRayTracer, ParaxialVsRealComparison
from optimizers.advanced_real_tracing import AdvancedRealRayTracer


class OptimizationComparison:
    """Класс для сравнения оптимизации с разными методами трассировки"""
    
    def __init__(self):
        self.results = {}
        
    def create_datasets(self) -> Dict[int, Dict]:
        """Создание датасетов для разных систем"""
        datasets = {}
        
        # 2-линзовая система
        datasets[2] = DataSetHelper.create_dataset(
            count_linse=2,
            focus_0={1: 200, 2: 200},
            harmonica={1: 7, 2: 7.5},
            distance={f'1-2': 10}
        )
        
        # 3-линзовая система
        datasets[3] = DataSetHelper.create_dataset(
            count_linse=3,
            focus_0={1: 300, 2: 300, 3: 300},
            harmonica={1: 7, 2: 7, 3: 7},
            distance={f'1-2': 10, f'2-3': 10}
        )
        
        # 4-линзовая система
        datasets[4] = DataSetHelper.create_dataset(
            count_linse=4,
            focus_0={1: 400, 2: 400, 3: 400, 4: 400},
            harmonica={1: 7, 2: 7, 3: 7, 4: 7},
            distance={f'1-2': 10, f'2-3': 10, f'3-4': 10}
        )
        
        return datasets
    
    def optimize_with_paraxial(self, dataset: Dict, system_name: str) -> Tuple[List[float], float]:
        """Оптимизация с параксиальной трассировкой"""
        count_linse = dataset['count_linse']
        
        if count_linse == 2:
            optimizer = TwoLensOptimizer()
            h_range = np.linspace(5, 10, 100)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0]
            
            print(f"Оптимизация {system_name} (параксиальная)...")
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
            optimizer = ThreeLensOptimizer()
            h_range = np.linspace(5, 10, 100)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0, 7.0]
            
            print(f"Оптимизация {system_name} (параксиальная)...")
            for h2 in h_range:
                for h3 in h_range:
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
            optimizer = FourLensOptimizer()
            h_range = np.linspace(5, 10, 50)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0, 7.0, 7.0]
            
            print(f"Оптимизация {system_name} (параксиальная)...")
            for h2 in h_range:
                for h3 in h_range:
                    for h4 in h_range:
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
            optimizer = FiveLensOptimizer()
            h_range = np.linspace(5, 10, 50)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0, 7.0, 7.0, 7.0]

            print(f"Оптимизация {system_name} (параксиальная)...")
            for h2 in h_range:
                for h3 in h_range:
                    for h4 in h_range:
                        for h5 in h_range:
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

        else:
            raise ValueError(f"Unsupported number of lenses: {count_linse}")
        
        print(f"Параксиальная оптимизация завершена. Мин фок. отрезок: {min_foc_dist*1000:.4f} мм")
        return best_heights, min_foc_dist
    
    def evaluate_with_real_tracing(self, dataset: Dict, heights: List[float], 
                                 system_name: str) -> Dict:
        """Оценка найденных высот с помощью реальной трассировки"""
        count_linse = dataset['count_linse']
        
        # Создаём реальный трассировщик
        real_tracer = RealRayTracer(dataset)
        
        # Диапазон длин волн
        lambda_range = np.linspace(dataset['lower_lambda'] * 1e-9, 
                                 dataset['upper_lambda'] * 1e-9, 50)
        
        focus_positions = []
        
        print(f"Оценка {system_name} с реальной трассировкой...")
        
        for lmbd in lambda_range:
            focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
            # Используем параксиальный фокус для оценки
            focus_pos = focus_info['paraxial_focus']
            focus_positions.append(focus_pos)
        
        focus_positions = np.array(focus_positions)
        real_foc_dist = np.max(focus_positions) - np.min(focus_positions)
        
        # Также оценим аберрацию
        aberrations = []
        for lmbd in lambda_range:
            focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
            aberration = abs(focus_info['longitudinal_SA'])
            aberrations.append(aberration)
        
        avg_aberration = np.mean(aberrations)
        
        results = {
            'lambda_range': lambda_range * 1e9,  # В нм
            'focus_positions': focus_positions * 1000,  # В мм
            'real_foc_dist': real_foc_dist * 1000,  # В мм
            'avg_aberration': avg_aberration * 1000,  # В мм
            'heights': heights
        }
        
        print(f"Реальная трассировка завершена. Фок. отрезок: {real_foc_dist*1000:.4f} мм, Средняя аберрация: {avg_aberration*1000:.4f} мм")
        
        return results
    
    def optimize_with_real_tracing(self, dataset: Dict, system_name: str) -> Tuple[List[float], float]:
        """Прямая оптимизация с реальной трассировкой"""
        count_linse = dataset['count_linse']
        real_tracer = RealRayTracer(dataset)
        
        if count_linse == 2:
            h_range = np.linspace(5, 10, 30)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0]
            
            print(f"Прямая оптимизация {system_name} (реальная трассировка)...")
            for h1 in h_range:
                for h2 in h_range:
                    try:
                        results = self.evaluate_with_real_tracing(dataset, [h1, h2], f"2-линзовая (реальная)")
                        foc_dist = results['real_foc_dist']
                        
                        if foc_dist < min_foc_dist:
                            min_foc_dist = foc_dist
                            best_heights = [h1, h2]
                            
                    except Exception as e:
                        continue
                        
        elif count_linse == 3:
            h_range = np.linspace(5, 10, 15)
            min_foc_dist = float('inf')
            best_heights = [7.0, 7.0, 7.0]
            
            print(f"Прямая оптимизация {system_name} (реальная трассировка)...")
            for h1 in h_range:
                for h2 in h_range:
                    for h3 in h_range:
                        try:
                            results = self.evaluate_with_real_tracing(dataset, [h1, h2, h3], f"3-линзовая (реальная)")
                            foc_dist = results['real_foc_dist']
                            
                            if foc_dist < min_foc_dist:
                                min_foc_dist = foc_dist
                                best_heights = [h1, h2, h3]
                                
                        except Exception:
                            continue
        else:
            raise ValueError(f"Direct optimization not implemented for {count_linse} lenses")
        
        print(f"Прямая оптимизация с реальной трассировкой завершена. Мин фок. отрезок: {min_foc_dist:.4f} мм")
        return best_heights, min_foc_dist
    
    def run_comparison(self):
        """Запуск полного сравнения"""
        print("="*80)
        print("СРАВНЕНИЕ ОПТИМИЗАЦИИ С ПАРАКСИАЛЬНОЙ И РЕАЛЬНОЙ ТРАССИРОВКОЙ")
        print("="*80)
        
        datasets = self.create_datasets()
        systems = [2, 3, 4]
        
        for count_linse in systems:
            print(f"\n{'='*60}")
            print(f"СИСТЕМА ИЗ {count_linse} ЛИНЗ")
            print(f"{'='*60}")
            
            dataset = datasets[count_linse]
            system_name = f"{count_linse}-линзовая система"
            
            # 1. Оптимизация с параксиальной трассировкой
            print(f"\n1. Оптимизация с параксиальной трассировкой:")
            paraxial_heights, paraxial_foc_dist = self.optimize_with_paraxial(dataset, system_name)
            
            # 2. Оценка параксиальных высот с реальной трассировкой
            print(f"\n2. Оценка параксиальных высот с реальной трассировкой:")
            real_eval_results = self.evaluate_with_real_tracing(dataset, paraxial_heights, system_name)
            
            # 3. Прямая оптимизация с реальной трассировкой (только для 2 и 3 линз)
            if count_linse <= 3:
                print(f"\n3. Прямая оптимизация с реальной трассировкой:")
                real_opt_heights, real_opt_foc_dist = self.optimize_with_real_tracing(dataset, system_name)
                
                # Оценка реальных оптимальных высот с параксиальной трассировкой
                print(f"\n4. Оценка реальных оптимальных высот с параксиальной трассировкой:")
                if count_linse == 2:
                    optimizer = TwoLensOptimizer()
                    lmbd_f_dict = optimizer.lmbd_focus_dict(
                        dataset=dataset, 
                        heights={UnitType.MICROMETER: real_opt_heights}, 
                        return_dict=True
                    )
                    paraxial_eval_foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict) * 1000
                elif count_linse == 3:
                    optimizer = ThreeLensOptimizer()
                    lmbd_f_dict = optimizer.lmbd_focus_dict(
                        dataset=dataset, 
                        heights={UnitType.MICROMETER: real_opt_heights}, 
                        return_dict=True
                    )
                    paraxial_eval_foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict) * 1000
                
                # Сохранение результатов
                self.results[count_linse] = {
                    'paraxial_opt': {
                        'heights': paraxial_heights,
                        'foc_dist': paraxial_foc_dist * 1000,
                        'real_eval_foc_dist': real_eval_results['real_foc_dist'],
                        'avg_aberration': real_eval_results['avg_aberration']
                    },
                    'real_opt': {
                        'heights': real_opt_heights,
                        'foc_dist': real_opt_foc_dist,
                        'paraxial_eval_foc_dist': paraxial_eval_foc_dist
                    }
                }
                
                # Вывод результатов
                print(f"\nРЕЗУЛЬТАТЫ ДЛЯ {count_linse}-ЛИНЗОВОЙ СИСТЕМЫ:")
                print(f"Параксиальная оптимизация:")
                print(f"  Оптимальные высоты: {[f'{h:.2f}' for h in paraxial_heights]} мкм")
                print(f"  Фокальный отрезок (паракс.): {paraxial_foc_dist*1000:.4f} мм")
                print(f"  Фокальный отрезок (реальный): {real_eval_results['real_foc_dist']:.4f} мм")
                print(f"  Средняя сферическая аберрация: {real_eval_results['avg_aberration']:.4f} мм")
                
                print(f"\nРеальная оптимизация:")
                print(f"  Оптимальные высоты: {[f'{h:.2f}' for h in real_opt_heights]} мкм")
                print(f"  Фокальный отрезок (реальный): {real_opt_foc_dist:.4f} мм")
                print(f"  Фокальный отрезок (паракс.): {paraxial_eval_foc_dist:.4f} мм")
                
                # Разница в фокальных отрезках
                diff_paraxial = abs(paraxial_foc_dist*1000 - real_eval_results['real_foc_dist'])
                diff_real = abs(real_opt_foc_dist - paraxial_eval_foc_dist)
                
                print(f"\nРАЗЛИЧИЯ:")
                print(f"  Параксиальная оптимизация даёт ошибку: {diff_paraxial:.4f} мм")
                print(f"  Реальная оптимизация даёт ошибку: {diff_real:.4f} мм")
                
                # Визуализация
                self.plot_comparison(count_linse, dataset)
                
            else:
                # Для 4-линзовой системы только параксиальная оптимизация
                self.results[count_linse] = {
                    'paraxial_opt': {
                        'heights': paraxial_heights,
                        'foc_dist': paraxial_foc_dist * 1000,
                        'real_eval_foc_dist': real_eval_results['real_foc_dist'],
                        'avg_aberration': real_eval_results['avg_aberration']
                    }
                }
                
                print(f"\nРЕЗУЛЬТАТЫ ДЛЯ {count_linse}-ЛИНЗОВОЙ СИСТЕМЫ:")
                print(f"Параксиальная оптимизация:")
                print(f"  Оптимальные высоты: {[f'{h:.2f}' for h in paraxial_heights]} мкм")
                print(f"  Фокальный отрезок (паракс.): {paraxial_foc_dist*1000:.4f} мм")
                print(f"  Фокальный отрезок (реальный): {real_eval_results['real_foc_dist']:.4f} мм")
                print(f"  Средняя сферическая аберрация: {real_eval_results['avg_aberration']:.4f} мм")
    
    def plot_comparison(self, count_linse: int, dataset: Dict):
        """Построение графиков сравнения"""
        result = self.results[count_linse]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Сравнение фокальных отрезков
        methods = ['Параксиальная оптимизация', 'Реальная оптимизация']
        paraxial_values = [result['paraxial_opt']['foc_dist'], result['real_opt']['paraxial_eval_foc_dist']]
        real_values = [result['paraxial_opt']['real_eval_foc_dist'], result['real_opt']['foc_dist']]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax1.bar(x - width/2, paraxial_values, width, label='Параксиальный расчёт', alpha=0.8)
        ax1.bar(x + width/2, real_values, width, label='Реальная трассировка', alpha=0.8)
        
        ax1.set_xlabel('Метод оптимизации')
        ax1.set_ylabel('Фокальный отрезок, мм')
        ax1.set_title(f'Сравнение фокальных отрезков ({count_linse} линзы)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for i, (p, r) in enumerate(zip(paraxial_values, real_values)):
            ax1.text(i - width/2, p + 0.01, f'{p:.3f}', ha='center', va='bottom')
            ax1.text(i + width/2, r + 0.01, f'{r:.3f}', ha='center', va='bottom')
        
        # График 2: Сравнение оптимальных высот
        ax2.set_title(f'Оптимальные высоты микрорельефа ({count_linse} линзы)')
        ax2.set_xlabel('Номер линзы')
        ax2.set_ylabel('Высота, мкм')
        
        lens_numbers = np.arange(1, count_linse + 1)
        paraxial_heights = result['paraxial_opt']['heights']
        real_heights = result['real_opt']['heights']
        
        ax2.plot(lens_numbers, paraxial_heights, 'bo-', linewidth=2, label='Параксиальная оптимизация')
        ax2.plot(lens_numbers, real_heights, 'ro-', linewidth=2, label='Реальная оптимизация')
        
        ax2.set_xticks(lens_numbers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Добавление значений
        for i, (p, r) in enumerate(zip(paraxial_heights, real_heights)):
            ax2.text(i + 1, p + 0.1, f'{p:.2f}', ha='center', va='bottom')
            ax2.text(i + 1, r + 0.1, f'{r:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        # Создаём папку если её нет
        import os
        os.makedirs(f'results/{count_linse}_linse', exist_ok=True)
        plt.savefig(f'results/{count_linse}_linse/optimization_comparison_{count_linse}_lenses.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Графики сравнения сохранены в results/{count_linse}_linse/optimization_comparison_{count_linse}_lenses.png")
    
    def print_summary(self):
        """Вывод итогового отчёта"""
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ОТЧЁТ О СРАВНЕНИИ ОПТИМИЗАЦИИ")
        print("="*80)
        
        for count_linse, result in self.results.items():
            print(f"\n{count_linse}-ЛИНЗОВАЯ СИСТЕМА:")
            
            if 'real_opt' in result:
                paraxial_opt = result['paraxial_opt']
                real_opt = result['real_opt']
                
                print(f"  Параксиальная оптимизация:")
                print(f"    Высоты: {[f'{h:.2f}' for h in paraxial_opt['heights']]} мкм")
                print(f"    Фок. отрезок (паракс.): {paraxial_opt['foc_dist']:.4f} мм")
                print(f"    Фок. отрезок (реальный): {paraxial_opt['real_eval_foc_dist']:.4f} мм")
                print(f"    Средняя аберрация: {paraxial_opt['avg_aberration']:.4f} мм")
                
                print(f"  Реальная оптимизация:")
                print(f"    Высоты: {[f'{h:.2f}' for h in real_opt['heights']]} мкм")
                print(f"    Фок. отрезок (реальный): {real_opt['foc_dist']:.4f} мм")
                print(f"    Фок. отрезок (паракс.): {real_opt['paraxial_eval_foc_dist']:.4f} мм")
                
                # Эффективность
                improvement = paraxial_opt['real_eval_foc_dist'] - real_opt['foc_dist']
                improvement_percent = (improvement / paraxial_opt['real_eval_foc_dist']) * 100
                
                print(f"  Улучшение реальной оптимизации:")
                print(f"    Снижение фок. отрезка: {improvement:.4f} мм ({improvement_percent:.2f}%)")
                
            else:
                paraxial_opt = result['paraxial_opt']
                print(f"  Параксиальная оптимизация:")
                print(f"    Высоты: {[f'{h:.2f}' for h in paraxial_opt['heights']]} мкм")
                print(f"    Фок. отрезок (паракс.): {paraxial_opt['foc_dist']:.4f} мм")
                print(f"    Фок. отрезок (реальный): {paraxial_opt['real_eval_foc_dist']:.4f} мм")
                print(f"    Средняя аберрация: {paraxial_opt['avg_aberration']:.4f} мм")
                print(f"    Примечание: Реальная оптимизация не проводилась (слишком много параметров)")


def main():
    """Основная функция"""
    comparator = OptimizationComparison()
    comparator.run_comparison()
    comparator.print_summary()


if __name__ == '__main__':
    main()
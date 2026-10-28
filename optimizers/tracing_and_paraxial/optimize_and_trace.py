"""
Скрипт для оптимизации параметров линз (параксиальный расчёт) и проверки
найденных высот трассировкой лучей (перенос C++ программы, пакет tracing)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from optimizers.four_lens import FourLensOptimizer
from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from utils.keys import DataSetKeys
from tracing import trace_focus_dict


class OptimizationAndTracing:
    """Класс для оптимизации и трассировки"""
    
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
    
    def optimize_two_lens(self, dataset: Dict,
                          h0 : float = 7.,
                          h_range : np.ndarray = None) -> Tuple[List[float], float]:
        """Оптимизация 2-линзовой системы"""
        optimizer = TwoLensOptimizer()
        
        # Поиск оптимальных высот
        if dataset is None:
            dataset = DataSetHelper.create_dataset(count_linse=2)
        h_range = np.linspace(5, 10, 100) if h_range is None else h_range
        min_foc_dist = float('inf')
        best_heights = [h0, h0]

        print("Оптимизация 2-линзовой системы...")
        
        start_time = time.time()
        for h2 in h_range:
            try:
                lmbd_f_dict = optimizer.lmbd_focus_dict(
                    dataset=dataset, 
                    heights={UnitType.MICROMETER: [h0, h2]}, 
                    return_dict=True
                )
                foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                
                if foc_dist < min_foc_dist:
                    min_foc_dist = foc_dist
                    best_heights = [h0, h2]
                    
            except Exception as e:
                continue
        
        end_time = time.time()

        print(f"Оптимизация 2х линз завершена за {(end_time - start_time) // 60} мин {(end_time - start_time) % 60:.2f} сек")
        print(f"Лучшие параметры: h1={best_heights[0]:.2f} мкм, h2={best_heights[1]:.2f} мкм")
        print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")
        
        DataSetHelper.to_json(dataset)

        return best_heights, min_foc_dist
    
    def optimize_three_lens(self, dataset: Dict,
                            h0: float = 7.,
                            h_range: np.ndarray = None) -> Tuple[List[float], float]:
        """Оптимизация 3-линзовой системы"""
        optimizer = ThreeLensOptimizer()
        
        # Поиск оптимальных высот
        if dataset is None:
            dataset = DataSetHelper.create_dataset(count_linse=3)
        h_range = np.linspace(5, 10, 100) if h_range is None else h_range
        min_foc_dist = float('inf')
        best_heights = [h0, h0, h0]
        
        print("Оптимизация 3-линзовой системы...")
        
        start_time = time.time()

        # Упрощённый перебор для ускорения
        for h2 in h_range:  # Шаг 2 для ускорения
            for h3 in h_range:
                    try:
                        lmbd_f_dict = optimizer.lmbd_focus_dict(
                            dataset=dataset, 
                            heights={UnitType.MICROMETER: [h0, h2, h3]}, 
                            return_dict=True
                        )
                        foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                        
                        if foc_dist < min_foc_dist:
                            min_foc_dist = foc_dist
                            best_heights = [h0, h2, h3]
                            
                    except Exception as e:
                        continue
        
        end_time = time.time()

        print(f"Оптимизация 3х линз завершена за {(end_time - start_time) // 60} мин {(end_time - start_time) % 60:.2f} сек")
        print(f"Лучшие параметры: h1={best_heights[0]:.2f} мкм, h2={best_heights[1]:.2f} мкм, h3={best_heights[2]:.2f} мкм")
        print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")

        DataSetHelper.to_json(dataset)
        
        return best_heights, min_foc_dist
    
    def optimize_four_lens(self, dataset: Dict,
                           h0: float = 7.,
                           h_range: np.ndarray = None) -> Tuple[List[float], float]:
        """Оптимизация 4-линзовой системы"""
        optimizer = FourLensOptimizer()
        
        # Поиск оптимальных высот (упрощённый перебор)
        if dataset is None:
            dataset = DataSetHelper.create_dataset(count_linse=4)
        h_range = np.linspace(5, 10, 50) if h_range is None else h_range
        min_foc_dist = float('inf')
        best_heights = [h0, h0, h0, h0]
        
        print("Оптимизация 4-линзовой системы...")

        start_time = time.time()
        
        # Очень упрощённый перебор для ускорения
        for h2 in h_range[::2]:
            for h3 in h_range[::2]:
                for h4 in h_range[::2]:
                    try:
                        lmbd_f_dict = optimizer.lmbd_focus_dict(
                            dataset=dataset, 
                            heights={UnitType.MICROMETER: [h0, h2, h3, h4]}, 
                            return_dict=True
                        )
                        foc_dist = optimizer.calc_focus_dist_static(lmbd_f_dict)
                        
                        if foc_dist < min_foc_dist:
                            min_foc_dist = foc_dist
                            best_heights = [h0, h2, h3, h4]
                            
                    except Exception as e:
                        continue

        end_time = time.time()
        
        print(f"Оптимизация 4х линз завершена за {(end_time - start_time) // 60} мин {(end_time - start_time) % 60:.2f} сек")
        print(f"Лучшие параметры: h1={best_heights[0]:.2f} мкм, h2={best_heights[1]:.2f} мкм, h3={best_heights[2]:.2f} мкм, h4={best_heights[3]:.2f} мкм")
        print(f"Минимальный фокальный отрезок: {min_foc_dist*1000:.4f} мм")

        DataSetHelper.to_json(dataset)
        
        return best_heights, min_foc_dist
    
    def trace_rays(self, dataset: Dict, heights: List[float], system_name: str) -> Dict:
        """
        Трассировка лучей для найденных оптимальных параметров: положение
        фокуса за последней линзой, усреднённое по вееру лучей (как N12Click в C++)
        """
        # Те же длины волн, что и в параксиальном расчёте
        lambda_range = np.linspace(dataset['lower_lambda'] * 1e-9,
                                 dataset['upper_lambda'] * 1e-9, 601)

        print(f"Трассировка лучей для {system_name}...")

        focus_dict = trace_focus_dict(dataset,
                                      heights={UnitType.MICROMETER: heights},
                                      lambda_massive=lambda_range)

        # Вычисление ширины фокального отрезка
        focus_positions = np.array(list(focus_dict.values()))
        focus_width = np.max(focus_positions) - np.min(focus_positions)
        
        results = {
            'lambda_range': lambda_range * 1e9,  # В нанометрах
            'focus_positions': focus_positions * 1000,  # В миллиметрах
            'focus_width': focus_width * 1000,  # В миллиметрах
            'heights': heights
        }
        
        print(f"Ширина фокального отрезка методом трассировки: {focus_width*1000:.4f} мм")
        
        return results
    
    def run_optimization_and_tracing(self):
        """Основной метод для запуска оптимизации и трассировки"""
        datasets = self.create_datasets()
        
        systems = [2, 3, 4]
        
        for count_linse in systems:
            print(f"\n{'='*50}")
            print(f"ОПТИМИЗАЦИЯ {count_linse}-ЛИНЗОВОЙ СИСТЕМЫ")
            print(f"{'='*50}")
            
            dataset = datasets[count_linse]
            
            # Оптимизация

            match count_linse:
                case 2: best_heights, min_foc_dist = self.optimize_two_lens(dataset)
                case 3: best_heights, min_foc_dist = self.optimize_three_lens(dataset)
                case 4: best_heights, min_foc_dist = self.optimize_four_lens(dataset)
                case _: raise ValueError("Unsupported number of lenses")

            # Трассировка
            system_name = f"{count_linse}-линзовая система"
            tracing_results = self.trace_rays(dataset, best_heights, system_name)
            
            # Сохранение результатов
            self.results[count_linse] =\
            {
                'dataset': dataset,
                'best_heights': best_heights,
                'min_foc_dist_paraxial': min_foc_dist,
                'tracing_results': tracing_results
            }
            
            # Визуализация
            self.plot_results(count_linse)
    
    def plot_results(self, count_linse: int):
        """Построение графиков результатов"""
        result = self.results[count_linse]

        save_path_dict = \
        {
            2: 'two_linse',
            3: 'three_linse',
            4: 'four_linse'
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Зависимость фокусного расстояния от длины волны (параксиальный расчёт)
        optimizer_class = {
            2: TwoLensOptimizer,
            3: ThreeLensOptimizer, 
            4: FourLensOptimizer
        }[count_linse]
        
        optimizer = optimizer_class()
        lmbd_f_dict = optimizer.lmbd_focus_dict(
            dataset=result['dataset'],
            heights={UnitType.MICROMETER: result['best_heights']},
            return_dict=True
        )
        
        lmbd_list = np.array(list(lmbd_f_dict.keys())) * 1e9  # В нм
        f_list = np.array(list(lmbd_f_dict.values())) * 1e3   # В мм
        
        ax1.plot(lmbd_list, f_list, 'b-', linewidth=2, label='Параксиальный расчёт')
        ax1.set_xlabel('Длина волны, нм')
        ax1.set_ylabel('Задний фокальный отрезок, мм')
        ax1.set_title(f'Параксиальный расчёт ({count_linse} линзы)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: Положение фокуса от длины волны (трассировка)
        tracing = result['tracing_results']
        ax2.plot(tracing['lambda_range'], tracing['focus_positions'], 'r-', linewidth=2, label='Трассировка лучей')
        ax2.set_xlabel('Длина волны, нм')
        ax2.set_ylabel('Положение фокуса, мм')
        ax2.set_title(f'Трассировка лучей ({count_linse} линзы)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Добавление информации о результатах
        textstr = f'Оптимальные высоты: {[f"{h:.2f}" for h in result["best_heights"]]} мкм\n' \
                 f'Фок. отрезок (паракс.): {result["min_foc_dist_paraxial"]*1000:.4f} мм\n' \
                 f'Фок. отрезок (трасс.): {tracing["focus_width"]:.4f} мм'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'results/{save_path_dict[count_linse]}/focus_analysis_{count_linse}_lenses.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Графики сохранены в results/{save_path_dict[count_linse]}/focus_analysis_{count_linse}_lenses.png")

    def start(self):
        print("Запуск оптимизации и трассировки для 2х, 3х и 4х линзовых систем...")
    
        self.run_optimization_and_tracing()
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ И ТРАССИРОВКИ")
        print("="*60)
        
        for count_linse, result in self.results.items():
            print(f"\n{count_linse}-линзовая система:")
            print(f"  Оптимальные высоты: {[f'{h:.2f} мкм' for h in result['best_heights']]}")
            print(f"  Фокальный отрезок (параксиальный): {result['min_foc_dist_paraxial']*1000:.4f} мм")
            print(f"  Фокальный отрезок (трассировка): {result['tracing_results']['focus_width']:.4f} мм")
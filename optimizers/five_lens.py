"""
Модуль для оптимизации 5-линзовой системы
"""

from typing import Dict, List, Tuple, Union

import numpy as np

from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from utils.keys import DataSetKeys
from optimizers.base import BaseLensOptimizer


class FiveLensOptimizer(BaseLensOptimizer):
    """Оптимизатор для 5-линзовой системы"""
    
    def __init__(self):
        super().__init__()
        self.count_linse = 5
        
    def create_dataset(self, focus_0: Dict[int, float], 
                      harmonica: Dict[int, float],
                      distance: Dict[str, float]) -> Dict[DataSetKeys, any]:
        """Создание датасета для 5-линзовой системы"""
        return DataSetHelper.create_dataset(
            count_linse=5,
            focus_0=focus_0,
            harmonica=harmonica,
            distance=distance
        )
    
    def lmbd_focus_dict(self, dataset: Dict[DataSetKeys, any],
                       heights: Dict[Union[UnitType, str], List[float]] = None,
                       lambda_massive: List[float] = None,
                       return_dict: bool = True) -> Dict[float, float]:
        """Расчёт задних фокальных отрезков для 5-линзовой системы (общая модель из base.py)"""
        if dataset['count_linse'] != 5:
            raise ValueError(f"Система должна состоять из 5 линз, а не {dataset['count_linse']}")

        if heights is not None:
            if isinstance(list(heights.keys())[0], str):
                heights = {UnitType.MICROMETER: heights['heights']}
            if len(list(heights.values())[0]) != 5:
                raise ValueError("Для 5-линзовой системы необходимо 5 высот")

        return super().lmbd_focus_dict(dataset, heights, lambda_massive, return_dict)
    
    def calc_focus_dist_static(self, lmbd_f_dict: Dict[float, float]) -> float:
        """Расчёт фокусного расстояния для 5-линзовой системы"""
        return super().calc_focus_dist_static(lmbd_f_dict)
    
    def optimize(self, dataset: Dict[DataSetKeys, any], 
                 h0: float = 7.0,
                 h_range: List[float] = None) -> Tuple[List[float], float]:
        """Оптимизация 5-линзовой системы"""
        if h_range is None:
            h_range = np.linspace(5, 10, 50)
        
        best_heights = [7.0] * 5
        min_foc_dist = float('inf')
        
        print("Оптимизация 5-линзовой системы...")
        
        # Упрощённый перебор для демонстрации
        # В реальной системе можно использовать более сложные методы оптимизации
        h_values = [h0, h0, h0, h0, h0] if h_range is None else h_range
        
        for h2 in h_range:
            for h3 in h_range:
                for h4 in h_range:
                    for h5 in h_range:
                        try:
                            lmbd_f_dict = self.lmbd_focus_dict(
                                dataset=dataset, 
                                heights={UnitType.MICROMETER: [h0, h2, h3, h4, h5]}, 
                                return_dict=True
                            )
                            foc_dist = self.calc_focus_dist_static(lmbd_f_dict)
                            
                            if foc_dist < min_foc_dist:
                                min_foc_dist = foc_dist
                                best_heights = [h0, h2, h3, h4, h5]
                                
                        except Exception:
                            continue
        
        return best_heights, min_foc_dist


if __name__ == '__main__':
    optimizer = FiveLensOptimizer()
    
    # Создаём датасет для 5-линзовой системы
    dataset = optimizer.create_dataset(
        focus_0={1: 500, 2: 500, 3: 500, 4: 500, 5: 500},
        harmonica={1: 7, 2: 7, 3: 7, 4: 7, 5: 7},
        distance={'1-2': 10, '2-3': 10, '3-4': 10, '4-5': 10}
    )
    
    # Оптимизация
    best_heights, min_foc_dist = optimizer.optimize(dataset)
    
    print(f"Оптимальные высоты: {best_heights}")
    print(f"Минимальный фокальный отрезок: {min_foc_dist * 1000:.4f} мм")
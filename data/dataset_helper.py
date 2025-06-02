from typing import TypedDict

from utils.keys import *
from utils.unit import UnitType, set_default_unit
from validators.input_validator import InputValidator

class DataSetConfig(TypedDict):
    count_linse: int
    lower_lambda: float
    upper_lambda: float
    refractive_index: dict[str, float]
    harmonica: dict[int, float]
    distance: dict[str, float]
    refractive_area: dict[str, float]
    lambda_0: dict[int, float]
    focus_0: dict[int, float]
    unit: dict[UnitKeys, UnitType]



class DataSetHelper:
    """Вспомгательный класс для создания датасетов"""


    @staticmethod
    def create_dataset(**kwargs) -> DataSetConfig:
        """
            Создает словарь по поданным значениям.\n
            Если какого-то необходимого ключа нету, то заполнится \n
            значение по умолчанию для этого ключа
        """

        if 'count_linse' not in kwargs: 
            raise KeyError("Не задано кол-во линз")
        
        expected_types = \
        {
            'count_linse' : int,
            'lower_lambda': float | int,
            'upper_lambda': float | int,
            'refractive_index': dict,
            'harmonica': dict,
            'distance': dict,
            'refractive_area': dict,
            'lambda_0': dict,
            'focus_0': dict,
            'unit': dict
        }
        
        count_linse = kwargs.get('count_linse')
        default_dataset = DataSetHelper.create_default_dataset(count_linse)
        
        for key, value in kwargs.items():
            if not isinstance(value, expected_types[key]):
                raise TypeError(f"{key} должен быть типа {expected_types[key].__name__}")
            else:
                default_dataset[key] = value

        InputValidator.validate_input_dataset(default_dataset)

        return default_dataset

    
    @staticmethod
    def create_default_dataset(count_linse : int) -> DataSetConfig:
        dataset : DataSetConfig = {}

        dataset['count_linse'] = count_linse
        dataset['lower_lambda'] = 400
        dataset['upper_lambda'] = 1000
        dataset['harmonica'] = \
        {
            i : 7
            for i in range(1, count_linse + 1)
        }
        dataset['distance'] = \
        {
            f'{i}-{i+1}' : 10
            for i in range(1, count_linse)
        }
        dataset['focus_0'] = \
        {
            i : 100
            for i in range(1, count_linse + 1)
        }
        dataset['lambda_0'] = \
        {
            i : 550
            for i in range(1, count_linse + 1)
        }
        dataset['refractive_area'] = \
        {
            f'{i}-{i+1}' : 1.
            for i in range(1, count_linse)
        }
        dataset['refractive_index'] = \
        {
            i : 1.5
            for i in range(1, count_linse + 1)
        }
        dataset['unit'] = set_default_unit(return_unit_type=True)

        InputValidator.validate_input_dataset(dataset)

        return dataset
            
        

from utils.keys import DataSetKeys 
from utils.unit import UnitType


# Словаь по умолчанию для 2-х линз

DATA_SET_2 : dict[DataSetKeys, int | float | dict[int | str, float]] = \
    {
    'count_linse': 2,
    'lower_lambda': 400,
    'upper_lambda': 1000,
    'refractive_index': \
        {                       # Показатель преломления каждой линзы 
            1: 1.5,
            2: 1.5
        },
    'harmonica' : \
        {                       # Набор гармоник
            1: 7,
            2: 7
        },
    'distance': \
        {                       # Расстояние м/у линзами в
            '1-2': 10
        },
    'refractive_area' : \
        {                       # Показатели преломления пространства
                                # м/у линзами
            '1-2': 1.
        },
    'lambda_0': \
        {                       # Базовая длина волны для каждой линзы
            1: 550,
            2: 550
        },
    'focus_0': \
        {                       # Базовый фокус для каждой линзы
            1: 200,
            2: 200
        },
    'unit' : \
        {
            'lower_lambda' : UnitType.NANOMETER,
            'upper_lambda' : UnitType.NANOMETER,
            'distance' : UnitType.MILLIMETER,
            'lambda_0' : UnitType.NANOMETER,
            'focus_0' : UnitType.MILLIMETER
        }
    }



# Словаь по умолчанию для 3-х гармонических линз

DATA_SET_3 : dict[DataSetKeys, int | float | dict[int | str, float]] = \
    {
    'count_linse': 3,
    'lower_lambda': 400,
    'upper_lambda': 1000,
    'refractive_index': \
        {                       # Показатель преломления каждой линзы 
            1: 1.5,
            2: 1.5,
            3: 1.5
        },
    'harmonica' : \
        {                       # Набор гармоник
            1: 7,
            2: 7,
            3: 7
        },
    'distance': \
        {                       # Расстояние м/у линзами в [см]
            '1-2': 10,
            '2-3': 10
        },
    'refractive_area' : \
        {                       # Показатели преломления пространства
                                # м/у линзами
            '1-2': 1.,
            '2-3': 1.
        },
    'lambda_0': \
        {                       # Базовая длина волны для каждой линзы в [нм]
            1: 550,
            2: 550,
            3: 550
        },
    'focus_0': \
        {                       # Базовый фокус для каждой линзы в [см]
            1: 300,
            2: 300,
            3: 300
        },
    'unit' : \
        {
            'lower_lambda' : UnitType.NANOMETER,
            'upper_lambda' : UnitType.NANOMETER,
            'distance' : UnitType.MILLIMETER,
            'lambda_0' : UnitType.NANOMETER,
            'focus_0' : UnitType.MILLIMETER
        }
    }
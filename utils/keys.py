from typing import Literal

# Разные типы ключей

SimpleKeys = Literal['h', 'd', 'f_0']
HSimpleKey = Literal['h']
DSimpleKey = Literal['d']
FSimpleKey = Literal['f_0']

ParamKeys = Literal['height', 'distance', 'focus_0']
HParamKey = Literal['height']
DParamKey = Literal['distance']
FParamKey = Literal['focus_0']

DataSetKeys = Literal['count_linse', 'lower_lambda', 'upper_lambda', 
                      'refractive_index', 'harmonica', 'distance',
                      'refractive_area', 'lambda_0', 'focus_0',
                      'unit']

UnitKeys = Literal['lower_lambda', 'upper_lambda', 'distance',
                   'lambda_0', 'focus_0', 'height']

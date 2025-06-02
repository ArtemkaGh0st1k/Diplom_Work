from enum import Enum
from typing import Optional

from utils.keys import *


class Unit():
    METER = 1
    CENTIMETR = 1e-2
    MILLIMETER = 1e-3
    MICROMETER = 1e-6
    NANOMETER = 1e-9
    PICOMETER = 1e-12 


class UnitType(Enum):
    METER = (Unit.METER, Unit.METER)
    CENTIMETR = (Unit.CENTIMETR, 1e2)
    MILLIMETER = (Unit.MILLIMETER, 1e3)
    MICROMETER = (Unit.MICROMETER, 1e6)
    NANOMETER = (Unit.NANOMETER, 1e9)
    PICOMETER = (Unit.PICOMETER, 1e12)


def set_default_unit(return_unit_type = False, 
                     unit_type : dict[UnitKeys, UnitType] = None) -> Optional[dict[UnitKeys, UnitType]]:
    
    if unit_type is None:
        unit_type = {}
        
    unit_type['height'] = UnitType.MICROMETER
    unit_type['distance'] = UnitType.MILLIMETER
    unit_type['focus_0'] = UnitType.MILLIMETER
    unit_type['lambda_0'] = UnitType.NANOMETER
    unit_type['lower_lambda'] = UnitType.NANOMETER
    unit_type['upper_lambda'] = UnitType.NANOMETER

    if return_unit_type: 
        return unit_type


def get_type_enum(key : str,
                  values : dict[int | float, UnitType]) -> tuple[SimpleKeys, UnitType]:
    '''
    Description:
    -----------
    Возвращает ключ и значение для корректировки шага по размерности
    '''

    key_res = None
    enum_res = None

    key_res = to_simple_key(key)

    for v in values:
        match v[1]:
            case UnitType.METER:
                enum_res = UnitType.METER
                break
            case UnitType.CENTIMETR:
                enum_res = UnitType.CENTIMETR
                break
            case UnitType.MILLIMETER:
                enum_res =  UnitType.MILLIMETER
                break
            case UnitType.MICROMETER:
                enum_res = UnitType.MICROMETER
                break
            case UnitType.NANOMETER:
                enum_res = UnitType.NANOMETER
                break
            case UnitType.PICOMETER:
                enum_res =  UnitType.PICOMETER
                break

    return (key_res, enum_res)


def unit_to_str_desc(unit : UnitType) -> str:
    match unit:
        case UnitType.METER:
            return 'м'
        case UnitType.CENTIMETR:
            return 'см'
        case UnitType.MICROMETER:
            return 'мкм'
        case UnitType.MILLIMETER:
            return 'мм'
        case UnitType.NANOMETER:
            return 'нм'
        case UnitType.PICOMETER:
            return 'пкм'
        case _:
            raise Exception('Не найден Unit')


def to_param_key(key : SimpleKeys) -> ParamKeys:
    match key:
        case 'd':
            return 'distance'
        case 'f_0':
            return 'focus_0'
        case 'h':
            return 'height'
        case _:
            raise KeyError('Ключ не найден')
        

def to_simple_key(key : ParamKeys) -> SimpleKeys:
    match key:
        case 'distance':
            return 'd'
        case 'focus_0':
            return 'f_0'
        case 'height':
            return 'h'
        case _:
            raise KeyError('Ключ не найден')
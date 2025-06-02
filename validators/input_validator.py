from typing import Any

from utils.keys import DataSetKeys


class InputValidator:
    """Проверяет корректность параметров оптимизатора"""
    
    
    @staticmethod
    def validate_input_dataset(dataset : dict[DataSetKeys, Any]):

        if dataset is None:
            raise ValueError("Пустой словарь!")
        
        ds_keys = list(dataset.keys())

        count_linse = 0
        count_ref_idx = 0
        count_harm = 0
        count_dist = 0
        count_ref_ar = 0
        count_lmbd0 = 0
        count_f0 = 0

        # проверка на наличие всех необходимых элементов в словаре

        dsk : DataSetKeys = None
        for dsk in DataSetKeys.__args__:
            if ds_keys.__contains__(dsk):
                match dsk:
                    case 'count_linse':
                        count_linse = dataset[dsk]
                    case 'refractive_index':
                        count_ref_idx = len(list(dataset[dsk].values()))
                    case 'harmonica':
                        count_harm = len(list(dataset[dsk].values()))
                    case 'distance':
                        count_dist = len(list(dataset[dsk].values()))
                    case 'refractive_area':
                        count_ref_ar = len(list(dataset[dsk].values()))
                    case 'lambda_0':
                        count_lmbd0 = len(list(dataset[dsk].values()))
                    case 'focus_0':
                        count_f0 = len(list(dataset[dsk].values()))
            else:
                raise KeyError("Не найден шаблонный ключ!")
            
        # для каждого ключа соблюдаеться строгое правило:
        # число параметров равно числу линз или на один меньше для некоторых
            
        if count_ref_idx < count_linse: raise IndexError("count_ref_idx < count_linse")
        if count_harm < count_linse: raise IndexError("count_harm < count_linse")
        if count_dist < count_linse - 1: raise IndexError("count_dist < count_linse - 1")
        if count_ref_ar < count_linse - 1: raise IndexError("count_ref_ar < count_linse - 1")
        if count_lmbd0 < count_linse: raise IndexError("count_lmbd0 < count_linse")
        if count_f0 < count_linse: raise IndexError("count_f0 < count_linse")



from operator import matmul
from functools import reduce
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data.datasets import DATA_SET_2, DATA_SET_3
from data.dataset_helper import DataSetConfig
from utils.converter import *
from utils.keys import *
from utils.unit import UnitType, unit_to_str_desc
from validators.input_validator import InputValidator


class BaseLensOptimizer:
    def __init__(self):

        self._lmbd_f_dict : dict = None
        self._len_f_dist : float = None
        self._dataset : dict[DataSetKeys, Any] = None

    
    @staticmethod
    def calc_focus_dist_static(lmbd_f_dict : dict[float, float]) -> float:
        f_list = list(lmbd_f_dict.values())
        return  max(f_list) - min(f_list)


    def calc_focus_dist(self):
        f_list = list(self._lmbd_f_dict.values())
        len_f_dist = max(f_list) - min(f_list)

        self._len_f_dist = len_f_dist
        return len_f_dist


    def lmbd_focus_dict(self,
                        dataset: dict[DataSetKeys, Any],
                        heights: dict[UnitType, list[float]] = None,
                        lambda_massive : list[float] = None,
                        return_dict = True) -> Optional[dict[float, float]]:
        
        '''
        Description:
        -----------
        Целевая функция\n
        Формулы и методы используемые в алгоритме:\n
        m -> гармоника, целое фиксированное число\n
        lambda_0 -> базовая длина волны, идёт подбор этой длины в зависимости от поданной высоты\n
        f = (m * lambda_0) / (k * lambda) * f_0 \n
        матрица преломления -> R = [1 0; -D 1], где D - оптич.сила\n
        матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами

        Return: 
        -------
        Возвращяет словарь длин волн и фокусных расстояний

        :Params:\n
        -------
        - dataset: Исходные данные в виде словаря\n
        - heights: Список высот и их размерность\n
        - lambda_massive: Массив высот\n
        - return_dict: Нужно ли вовзращать словарь
        '''

        try:
            InputValidator.validate_input_dataset(dataset)
        except (KeyError, IndexError, ValueError):
            dataset = DATA_SET_2.copy()     # используется словарь для 2-х линз по умолчанию !!!
            if heights is not None:
                if len(heights) == 2: dataset = DATA_SET_2.copy()
                if len(heights) == 3: dataset = DATA_SET_3.copy()
                else: raise Exception("Число высот должно быть равно 2 или 3!")
        except Exception:
            raise
        
        self._dataset = dataset.copy()

        h_list : list[float] = None
        h_unit : float = None
        if heights is not None:
            h_list = list(heights.values())[0]
            h_unit = list(heights.keys())[0].value[0]

        l_lmbd_unit = dataset['unit']['lower_lambda'].value[0]
        u_lmbd_unit = dataset['unit']['upper_lambda'].value[0]
        f0_unit = dataset['unit']['focus_0'].value[0]
        lmbd0_unit = dataset['unit']['lambda_0'].value[0]
        dist_unit = dataset['unit']['distance'].value[0]

        if lambda_massive is None:
            lambda_massive = np.linspace(dataset['lower_lambda'] * l_lmbd_unit,
                                        dataset['upper_lambda'] * u_lmbd_unit,
                                        601) 
        
        lmbd_f_dict = {}
        for lmbd in lambda_massive:
            matrix_mults_list = []

            for i in range(1, dataset['count_linse'] + 1):
                refractive_index = self.n_bk7(lmbd * 1e6)
                focus_0 = dataset['focus_0'][i] * f0_unit
                harmonica = dataset['harmonica'][i]
                
                height = (h_list[i-1] * h_unit
                          if heights is not None
                          else 
                          (dataset['harmonica'][i] * dataset['lambda_0'][i] / (dataset['refractive_index'][i] - 1)) * \
                           lmbd0_unit)
                #harmonica = custom_round(height * dataset['lambda_0'][i] / (refractive_index - 1) * 1e3) 
                
                lambda_0 = height * (refractive_index - 1) / harmonica            
                k = custom_round((lambda_0 / (lmbd)) * harmonica)
                if k == 0:
                    k = 0.5

                focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
                optic_power = (1 / focus)

                refractive_matrix = np.array\
                (
                    [
                        [1, 0],
                        [-optic_power, 1]
                    ]
                )

                matrix_mults_list.append(refractive_matrix)
                
                if i != dataset['count_linse']:

                    refractive_area = dataset['refractive_area']['{}-{}'.format(i, i+1)]
                    dist = dataset['distance']['{}-{}'.format(i, i+1)] * dist_unit

                    reduce_dist = dist / refractive_area

                    transfer_matrix = np.array\
                    (
                        [
                            [1, reduce_dist],
                            [0, 1]
                        ]
                    )

                    matrix_mults_list.append(transfer_matrix)
                        
            matrix_mults_list.reverse()
            matrix_mults_list = np.array(matrix_mults_list)

            mult_res = reduce(matmul, matrix_mults_list)
            lmbd_f_dict[lmbd] = - 1 / mult_res[1, 0]

        self._lmbd_f_dict = lmbd_f_dict
        self._len_f_dist = self.calc_focus_dist()
        if return_dict:
            return lmbd_f_dict


    def visualize_depend_f_lmbd(self,
                                units : dict[Literal['lmbd', 'f', 'foc_dist'], UnitType] = None,
                                return_fig_ax = False,
                                blockAndShow = True) -> tuple[Figure, Axes] | None:

        if units is None:
            units = {}
            units['lmbd'] = UnitType.NANOMETER
            units['f'] = UnitType.MILLIMETER
            units['foc_dist'] = UnitType.MILLIMETER
        if self._len_f_dist is None:
            raise Exception("Сначала посчитайте фок.отрезок т.к вызовете метод lmbd_focus_dict")

        lmbd_list = np.array(list(self._lmbd_f_dict.keys())) * units['lmbd'].value[1]
        f_list = np.array(list(self._lmbd_f_dict.values())) * units['f'].value[1]

        xl = unit_to_str_desc(units['lmbd'])
        yl = unit_to_str_desc(units['f'])
        leg_lbl = unit_to_str_desc(units['foc_dist'])

        ax : Axes = None
        fig, ax = plt.subplots(1, 1)

        ax.set_title('Зависимость фокусного расстояния от длины волны')
        ax.set_xlabel(f'Длина волны, {xl}')
        ax.set_ylabel(f'Фокусное расстояние, {yl}')
        ax.grid(which="both")
        ax.plot(lmbd_list, f_list)
        ax.legend(loc='upper center',
                    bbox_to_anchor=(0.5, 1.0),
                    title="Длина фокального отрезка: {} {}".format(round(self._len_f_dist * units['foc_dist'].value[1], 3), leg_lbl))
        if blockAndShow:
            plt.show(block=blockAndShow)
        plt.close()
        if return_fig_ax:
            return (fig, ax)


    def visualize_dummmy_loss(self,
                              dataset : DataSetConfig,
                              idxs : tuple[int, int] = (0, 1)):
        # для высот

        h1_range = np.linspace(3, 15, 50)
        h2_range = np.linspace(3, 15, 50)
        H1, H2 = np.meshgrid(h1_range, h2_range)

        loss_values = np.zeros_like(H1)

        idx1, idx2 = idxs[0], idxs[1]

        for i in range(H1.shape[0]):
            for j in range(H1.shape[1]):
                for k in range(H1.shape[0]):
                    self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h1_range[i], h2_range[j], h1_range[k]]})
                    loss_values[i, j] = (self.calc_focus_dist() ** 2)

        #ax : Axes = None
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(H1, H2, loss_values, cmap='plasma')
        ax.set_xlabel(f'h{idx1+1}')
        ax.set_ylabel(f'h{idx2+1}')
        ax.set_zlabel('Loss')
        ax.set_title('Loss Landscape with Multiple Wavelengths')
        plt.show(block=True)



    @staticmethod
    def merge_axes_static(ax1 : Axes, ax2 : Axes, foc_dists : list[float],
                          labels : list[str] = None,
                          colors : list[str] = ['red', 'blue'],
                          blockAndShow = True):
        foc_dists = np.array(foc_dists) * 1e3
        ax : Axes = None
        fig, ax = plt.subplots(1, 1)
        
        if labels is None:
            labels = []
            labels.append(f"Мин фок.отрезок для 1й-линзы {foc_dists[0]:.3f} мм")
            labels.append(f"Мин фок.отрезок для 2х-линз {foc_dists[1]:.3f} мм")
        else:
            labels[0] += f" {foc_dists[0]:.3f} мм"
            labels[1] += f" {foc_dists[1]:.3f} мм"

        for line1 in ax1.get_lines():
            ax.plot(line1.get_xdata(), line1.get_ydata(), color=colors[0], label=labels[0])
        
        for line2 in ax2.get_lines():
            ax.plot(line2.get_xdata(), line2.get_ydata(), color=colors[1], label=labels[1])
        
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("Фокусное расстояние, мм")
        ax.set_title("Сравнение фок отрезка")
        ax.grid(which='both')
        ax.legend()

        if blockAndShow:
            plt.show(block=True)
        plt.close()
        
        # тестирование некоторых случаев 

    
    def n_bk7(self, lambda_um: float) -> float:
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
        l2 = lambda_um**2
        n2 = 1 + (B1 * l2) / (l2 - C1) + (B2 * l2) / (l2 - C2) + (B3 * l2) / (l2 - C3)
        return n2**0.5

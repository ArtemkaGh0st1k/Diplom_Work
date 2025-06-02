from operator import matmul
from functools import reduce
from typing import Any, Optional, overload, Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data.datasets import DATA_SET_2, DATA_SET_3
from data.dataset_helper import DataSetConfig, DataSetHelper
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
    def calc_focus_dist_static(lmd_f_dict : dict) -> float:
        f_values = lmd_f_dict.values()
        return max(f_values) - min(f_values)
    

    def calc_focus_dist(self) -> float:
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
        - heights: Список высот и их размерность \n
        - lambda_massive: Массив длин волн\n
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
                refractive_index = dataset['refractive_index'][i]
                focus_0 = dataset['focus_0'][i] * f0_unit
                harmonica = dataset['harmonica'][i]
                
                #TODO: проверить случай: когда подаётся высота, то должна ли изменяться гаромника? 
                #TODO: хотя ниже пересчитывается базовая длина волны
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
                    self.lmbd_focus_dict(dataset,heights={UnitType.MICROMETER : [h1_range[i], h2_range[j], h1_range[k]]})
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
        plt.close()


    @overload
    def generate_grid_with_fixed_height(self, 
                                        init_h : list[float],
                                        hbounds : tuple[float, float],
                                        h_static : Literal['h1', 'h2'],
                                        m_list : list[float],
                                        func_list : list[Callable[[Any], float]],
                                        check_neighbour_min = False,
                                        unit_f_dist = UnitType.MILLIMETER,
                                        units : dict[UnitKeys, UnitType] = None) -> None:
        
        h_range = np.linspace(hbounds[0], hbounds[1], 300)
        foc_dist_list_anal = []
        foc_dist_list_matrix = []
        
        if h_static == 'h1':
            h_static_val = init_h[0]
        else:
            h_static_val = init_h[1]

        if func_list[0].__name__.__contains__("matrix"):
            matrix_func = func_list[0]
            anal_func = func_list[1]
        else:
            matrix_func = func_list[1]
            anal_func = func_list[0]

        for _, h in enumerate(h_range):
            foc_dist_list_anal.append(anal_func(h, h_static_val, m_list[0], m_list[1], h_static))
            foc_dist_list_matrix.append(matrix_func(h, h_static_val, m_list[0], m_list[1], h_static))

        unit_foc_dist_matrix = np.array(foc_dist_list_matrix) * unit_f_dist.value[1]
        min_foc_dist_matrix = min(unit_foc_dist_matrix)
        min_idx_matrix = np.argmin(unit_foc_dist_matrix)

        unit_foc_dist_anal = np.array(foc_dist_list_anal) * unit_f_dist.value[1]
        min_foc_dist_anal = min(unit_foc_dist_anal)
        min_idx_anal = np.argmin(unit_foc_dist_anal)

        find_h_matrix = h_range[min_idx_matrix]
        find_h_anal = h_range[min_idx_anal]

        axs : list[Axes] = None

        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        fig.suptitle(f'Поведение фок.отрезка от изменения высоты | Нач.приближение = {init_h[0]:.1f}, {init_h[1]:.1f}')

        titles = ['Матричный метод', 'Аналит.метод']
        xlabel = 'Высота {}й линзы, мкм'.format(h_static.replace('h', ''))
        ylabel = 'Фок. отрезок, {}'.format(unit_to_str_desc(unit_f_dist))
        plot_arg = [unit_foc_dist_matrix, unit_foc_dist_anal]
        scater_arg = [find_h_matrix, find_h_anal,
                      min_foc_dist_matrix, min_foc_dist_anal]
        annotate_arg = scater_arg

        if check_neighbour_min:

            # найдём предыдущий минимум для массива, чтобы показать точность нахождения гармоники
            unit_foc_dist_matrix_copy = np.delete(unit_foc_dist_matrix, min_idx_matrix)
            prev_min_foc_dist_matrix = min(unit_foc_dist_matrix_copy)
            prev_min_idx_matrix = np.argmin(unit_foc_dist_matrix_copy)
            prev_find_h_matrix = h_range[prev_min_idx_matrix]

            unit_foc_dist_anal_copy = np.delete(unit_foc_dist_anal, min_idx_anal)
            prev_min_foc_dist_anal = min(unit_foc_dist_anal_copy)
            prev_min_idx_anal = np.argmin(unit_foc_dist_anal_copy)
            prev_find_h_anal = h_range[prev_min_idx_anal]

            scater_arg.extend([prev_find_h_matrix, prev_find_h_anal, prev_min_foc_dist_matrix, prev_min_foc_dist_anal])

        for i, ax in enumerate(axs):
            ax.set_title(titles[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(which='both')
            ax.plot(h_range, plot_arg[i])
            ax.scatter(scater_arg[i], scater_arg[i+2], c='r', label='Мин фок.отрезок')
            ax.annotate(f'({annotate_arg[i]:.2f}, {annotate_arg[i+2]:.2f})',
                        xy=(annotate_arg[i], annotate_arg[i+2]),
                        xytext=(-10, -10),
                        textcoords='offset points',
                        ha='left', va='top',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
            
            if check_neighbour_min:
                ax.scatter(scater_arg[i+4], scater_arg[i+6], c='r')
                ax.annotate(f'({annotate_arg[i+4]:.2f}, {annotate_arg[i+6]:.2f})',
                            xy=(annotate_arg[i+4], annotate_arg[i+6]),
                            xytext=(-10, -10),
                            textcoords='offset points',
                            ha='right', va='bottom',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5),
                            arrowprops=dict(arrowstyle='->'))
            ax.legend()

        plt.show(block=True)
        plt.close()
    

    @overload
    def generate_grid_with_fixed_height(self,
                                        init_h : dict[UnitType, list[float]],
                                        hbounds : tuple[float, float], 
                                        dataset : dict[DataSetKeys, Any],
                                        h_static : Literal['h1', 'h2', 'h3', 'h4'] = 'h1',
                                        units : dict[UnitKeys, UnitType] = None,
                                        unit_f_dist : UnitType = UnitType.MILLIMETER):
        
        try:
            InputValidator.validate_input_dataset(dataset)
        except (IndexError, KeyError, ValueError):
            count_linse = dataset['count_linse']
            dataset = DataSetHelper.create_default_dataset(count_linse)
        except:
            raise

        h_range = np.linspace(hbounds[0], hbounds[1], 50)
                


    @overload
    def draw_grid(self):
        pass


    @overload
    def draw_grid(self):
        pass


    #TODO: доработать слияния осей
    def merge_axes(self, 
                   ax2 : Axes,
                   foc_dist : float,
                   labels : list[str] = None,
                   colors : list[str] = ['red', 'blue'],
                   blockAndShow = True):
        foc_dist *= 1e3
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


    def lmbd_focus_dict_recalc_m(self,
                        dataset: dict[DataSetKeys, Any],
                        heights: dict[UnitType, list[float]] = None,
                        lambda_massive : list[float] = None,
                        return_dict = True) -> Optional[dict[float, float]]:

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
                                        1000) 
        
        lmbd_f_dict = {}
        for lmbd in lambda_massive:
            matrix_mults_list = []

            for i in range(1, dataset['count_linse'] + 1):
                refractive_index = dataset['refractive_index'][i]
                focus_0 = dataset['focus_0'][i] * f0_unit
                lambda_0 = dataset['lambda_0'][i] * lmbd0_unit

                height = (h_list[i-1] * h_unit
                          if heights is not None
                          else 
                          (dataset['harmonica'][i] * dataset['lambda_0'][i] / (dataset['refractive_index'][i] - 1)) * \
                           lmbd0_unit)
                harmonica = custom_round(height * dataset['lambda_0'][i] / (refractive_index - 1) * 1e3)
                          
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
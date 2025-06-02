from typing import Any
from operator import matmul
from functools import reduce
import time
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from optimizers.base import BaseLensOptimizer
from utils.unit import UnitType, set_default_unit, unit_to_str_desc
from utils.keys import *
from utils.converter import *
from data.datasets import DATA_SET_2


class TwoLensOptimizer(BaseLensOptimizer):
    def __init__(self):
        super().__init__()


    def lmbd_focus_dict(self, dataset, heights = None, lambda_massive = None, return_dict=True):
        if dataset['count_linse'] != 2:
            raise Exception(f"Система должна состоять из 2=х линз, а не {self._dataset['count_linse']}")
        return super().lmbd_focus_dict(dataset, heights, lambda_massive, return_dict)
    

    def visualize_depend_f_lmbd(self, units = None, return_fig_ax=False, blockAndShow=True):
        if self._dataset['count_linse'] != 2:
            raise Exception(f"Система должна состоять из 2=х линз, а не {self._dataset['count_linse']}")
        return super().visualize_depend_f_lmbd(units, return_fig_ax, blockAndShow)
    

    def calc_focus_dist_2_lisne_analytical(self,
                                           h_optimize, h_static, m1, m2,
                                           h1_or_h2_optimize : Literal['h1', 'h2'],
                                           dataset : dict[DataSetKeys, Any] = None,
                                           units : dict[UnitKeys, UnitType] = None) -> float:
        
        if dataset is None:
            dataset = DATA_SET_2.copy()

        if units is None:
            units = set_default_unit(unit_type=units, return_unit_type=True)
        
        h_1 = None
        h_2 = None

        if h1_or_h2_optimize == 'h1':
            h_1 = h_optimize
            h_2 = h_static
        else:
            h_1 = h_static
            h_2 = h_optimize

        h_1 *= units['height'].value[0]
        h_2 *= units['height'].value[0]

        lmbd_range = np.linspace(dataset['lower_lambda'] * units['lower_lambda'].value[0],
                                 dataset['upper_lambda'] * units['upper_lambda'].value[0],
                                 1000)

        lmbd_f_dict = {}
        for lmbd in lmbd_range:

            f0_1 = dataset['focus_0'][1] * units['focus_0'].value[0]
            f0_2 = dataset['focus_0'][2] * units['focus_0'].value[0]

            ref_idx1 = dataset['refractive_index'][1]
            ref_idx2 = dataset['refractive_index'][2]

            ref_ar = dataset['refractive_area']['1-2']

            d = dataset['distance']['1-2'] * units['distance'].value[0] / ref_ar

            lmbd0_1 = h_1 * (ref_idx1 - 1) / m1
            lmbd0_2 = h_2 * (ref_idx2 - 1) / m2

            k1 = custom_round((lmbd0_1 / (lmbd)) * m1)
            k2 = custom_round((lmbd0_2 / (lmbd)) * m2)

            f1 = ((m1 * lmbd0_1) / (k1 * lmbd)) * f0_1
            f2 = ((m2 * lmbd0_2) / (k2 * lmbd)) * f0_2

            D1 = 1 / f1
            D2 = 1 / f2
            D = D1 + D2 - d * D1 * D2

            lmbd_f_dict[lmbd] = 1 / D

        len_f_dist = super().calc_focus_dist_static(lmbd_f_dict)
        self._len_f_dist = len_f_dist
        return len_f_dist
    

    def calc_focus_dist_2_linse_matrix(self,
                                       h_optimize, h_static, m1, m2,
                                       h1_or_h2_optimize : Literal['h1', 'h2'],
                                       dataset : dict[DataSetKeys, Any] = None,
                                       units : dict[UnitKeys, UnitType] = None) -> float:
        
        if dataset is None:
            dataset = DATA_SET_2.copy()

        if units is None:
            units = set_default_unit(return_unit_type=True)
        
        h_1 = None
        h_2 = None

        if h1_or_h2_optimize == 'h1':
            h_1 = h_optimize
            h_2 = h_static
        else:
            h_1 = h_static
            h_2 = h_optimize

        h_1 *= units['height'].value[0]
        h_2 *= units['height'].value[0]

        h_list = [h_1, h_2]
        harm_list = [m1, m2]

        lmbd_range = np.linspace(dataset['lower_lambda'] * units['lower_lambda'].value[0],
                                 dataset['upper_lambda'] * units['upper_lambda'].value[0],
                                 1000)
        
        lmbd_f_dict = {}
        for lmbd in lmbd_range:
            matrix_mults_list = []

            for i in range(1 , dataset['count_linse'] + 1):

                #harmonica = harm_list[i-1]

                lmbd0 = dataset['lambda_0'][i] * units['lambda_0']._value_[0]
                ref_idx = dataset['refractive_index'][i]
                focus_0 = dataset['focus_0'][i] * units['focus_0'].value[0]
                height = h_list[i-1]
                harmonica = height * (ref_idx - 1) / lmbd0 

                #lmbd0 = height * (ref_idx - 1) / harmonica

                k = custom_round((lmbd0 / (lmbd)) * harmonica)
                if k == 0:
                    k = 0.5

                focus = ((harmonica * lmbd0) / (k * lmbd)) * focus_0
                optic_power =  1 / focus

                refractive_matrix = np.array\
                (
                    [
                        [1, 0],
                        [-optic_power, 1]
                    ]
                )

                matrix_mults_list.append(refractive_matrix)

                if i != dataset['count_linse']:
                    ref_ar = dataset['refractive_area']['{}-{}'.format(i, i+1)]
                    dist = dataset['distance']['{}-{}'.format(i, i+1)] * units['distance'].value[0]

                    reduce_dist = dist / ref_ar

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

        len_f_dist = super().calc_focus_dist_static(lmbd_f_dict)
        self._len_f_dist = len_f_dist
        return len_f_dist
    

    def generate_grid_with_fixed_height(self, 
                                        init_h : list | np.ndarray,
                                        hbounds : tuple[float, float],
                                        m1 : float, m2 : float,
                                        h1_or_h2_optimize : Literal['h1', 'h2'] = 'h2',
                                        check_neighbour_min = False,
                                        units : dict[UnitKeys, UnitType] = None,
                                        unit_f_dist : UnitType = UnitType.MILLIMETER) -> None:
        
        h_range = np.linspace(hbounds[0], hbounds[1], 300)
        foc_dist_list_anal = []
        foc_dist_list_matrix = []

        h_optimize = None
        h_static = None

        if h1_or_h2_optimize == 'h1':
            h_static = init_h[1]
        else:
            h_static = init_h[0]

        for _, h in enumerate(h_range):
            h_optimize = h

            foc_dist_list_anal.append(self.calc_focus_dist_2_lisne_analytical(h_optimize, h_static, m1, m2, h1_or_h2_optimize))
            foc_dist_list_matrix.append(self.calc_focus_dist_2_linse_matrix(h_optimize, h_static, m1, m2, h1_or_h2_optimize))

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
        xlabel = 'Высота {}й линзы, мкм'.format(h1_or_h2_optimize.replace('h', ''))
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


    def generate_grid_without_fixed_height(self, 
                                           hbounds : tuple[float, float],
                                           step_h = 0.1,
                                           dataset : dict[DataSetKeys, Any] = None,
                                           units : dict[UnitKeys, UnitType] = None):
        
        if dataset is None:
            dataset = DATA_SET_2.copy()
        if units is None:
            set_default_unit(unit_type=units)

        h_range = np.arange(hbounds[0], hbounds[1] + step_h, step_h)
        
        foc_dist_h1_fixed = np.zeros((len(h_range), len(h_range)))    # фиксируем первую - перебираем вторую
        foc_dist_h2_fixed = np.zeros((len(h_range), len(h_range)))    # фиксируем вторую - перебираем первую

        start_time = time.time()
        for i, h1 in enumerate(h_range):
            for j, h2 in enumerate(h_range):
                lmbd_f_h1_fixed = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h1, h2]})
                lmbd_f_h2_fixed = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h2, h1]})

                foc_dist_h1_fixed[i, j] = super().calc_focus_dist_static(lmbd_f_h1_fixed)
                foc_dist_h2_fixed[i, j] = super().calc_focus_dist_static(lmbd_f_h2_fixed)

        print("Время работы алгоритма для 2-х линз: ",
              Fore.RED, f"{(time.time() - start_time) // 60} м",
              Fore.GREEN, f"{(time.time() - start_time) % 60} с",
              Fore.WHITE)

        # поиск минимума

        # для фиксируемой h1
        min_idx_fix_h1 = np.unravel_index(np.argmin(foc_dist_h1_fixed), foc_dist_h1_fixed.shape)
        min_h1_fix_h1 = h_range[min_idx_fix_h1[0]]
        min_h2_fix_h1 = h_range[min_idx_fix_h1[1]]
        min_foc_dist_fix_h1 = foc_dist_h1_fixed[min_idx_fix_h1]

        # для фиксируемой h2
        min_idx_fix_h2 = np.unravel_index(np.argmin(foc_dist_h2_fixed), foc_dist_h2_fixed.shape)
        min_h1_fix_h2 = h_range[min_idx_fix_h2[0]]
        min_h2_fix_h2 = h_range[min_idx_fix_h2[1]]
        min_foc_dist_fix_h2 = foc_dist_h1_fixed[min_idx_fix_h2]


        # визуализация 

        axs : list[Axes] = None

        fig, axs = plt.subplots(1, 2)
        plt.suptitle("Поиск мин фок. отрезка")
        
        imshow_arg = [foc_dist_h1_fixed, foc_dist_h2_fixed]
        scatter_arg = [min_h1_fix_h1, min_h1_fix_h2,
                       min_h2_fix_h1, min_h2_fix_h2,
                       min_foc_dist_fix_h1, min_foc_dist_fix_h2]
        xlabel = "h1, {}".format(unit_to_str_desc(units['height']))
        ylabel = "h2, {}".format(unit_to_str_desc(units['height']))
        titles = ["Перебор 2-й линзы", "Перебор 1-й линзы"]

        for i, ax in enumerate(axs):
            im = ax.imshow(imshow_arg[i], extent=[min(h_range), max(h_range), min(h_range), max(h_range)], cmap='viridis')
            plt.colorbar(im)
            ax.scatter(scatter_arg[i],
                       scatter_arg[i+2],
                       color='red',
                       label=f"Мин фок.отрезок в ({scatter_arg[i]:.2f}, {scatter_arg[i+2]:.2f}) = {scatter_arg[i+4]}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(titles[i])
            ax.legend()
            ax.grid(which='both')
            plt.tight_layout()
            
        plt.show(block=True)
        plt.close()



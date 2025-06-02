from typing import Literal, Any
import time
from colorama import Fore
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from optimizers.base import BaseLensOptimizer
from validators.input_validator import InputValidator
from data.datasets import DATA_SET_3
from utils.keys import *
from utils.unit import *


class ThreeLensOptimizer(BaseLensOptimizer):
    def __init__(self,):
        super().__init__()

        self._fig_ax : tuple[Figure, Axes] = None

    
    @property
    def get_fig_ax(self) -> tuple[Figure, Axes]:
        return self._fig_ax


    def lmbd_focus_dict(self, dataset, heights = None, lambda_massive = None, return_dict=True):
        if dataset['count_linse'] != 3:
            raise Exception(f"Система должна состоять из 3=х линз, а не {self._dataset['count_linse']}")
        return super().lmbd_focus_dict(dataset, heights, lambda_massive, return_dict)

    
    def visualize_depend_f_lmbd(self, units = None, return_fig_ax=False, blockAndShow=True):
        if self._dataset['count_linse'] != 3:
            raise Exception(f"Для визуализации система должна состоять из 3-х линз, а не {self._dataset['count_linse']}")
        return super().visualize_depend_f_lmbd(units, return_fig_ax, blockAndShow)


    def generate_grid_with_fixed_height(self, 
                                        init_h : dict[UnitType, list[float]],
                                        hbounds : tuple[float, float], 
                                        dataset : dict[DataSetKeys, Any],
                                        h_static : Literal['h1', 'h2', 'h3'] = 'h1',
                                        units : dict[UnitKeys, UnitType] = None,
                                        unit_f_dist : UnitType = UnitType.MILLIMETER) -> None:
    
        try:
            InputValidator.validate_input_dataset(dataset)
        except (IndexError, KeyError, ValueError):
            dataset = DATA_SET_3.copy()
        except Exception:
            raise
        
        h_range = np.linspace(hbounds[0], hbounds[1], 50)

        foc_dist_hi_optimize = np.zeros((len(h_range), len(h_range)))

        start_time = time.time()

        xlabel_i, xlabel_j = None, None
        ylabel_i, ylabel_j = None, None
        title = None
        
        h_static_val = None
        match h_static:
            case 'h1':
                h_static_val = list(init_h.values())[0][0]
                xlabel_i, xlabel_j = 'h3, мкм', 'h2, мкм'
                ylabel_i, ylabel_j = 'h2, мкм', 'h3, мкм'
                title = f'h1 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(h_range):
                    for j, h_j in enumerate(h_range):
                        lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_static_val, h_i, h_j]})
                        foc_dist_hi_optimize[i, j] = super().calc_focus_dist_static(lmbd_f_hi_optimize)

            case 'h2':
                h_static_val = list(init_h.values())[0][1]
                xlabel_i, xlabel_j = 'h3, мкм', 'h1, мкм'
                ylabel_i, ylabel_j = 'h1, мкм', 'h3, мкм'
                title = f'h2 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(h_range):
                    for j, h_j in enumerate(h_range):
                        lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_i, h_static_val, h_j]})
                        foc_dist_hi_optimize[i, j] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
                        
            case 'h3':
                h_static_val = list(init_h.values())[0][2]
                xlabel_i, xlabel_j = 'h2, мкм', 'h1, мкм'
                ylabel_i, ylabel_j = 'h1, мкм', 'h2, мкм'
                title = f'h3 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(h_range):
                    for j, h_j in enumerate(h_range):
                        lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_i, h_j, h_static_val]})
                        foc_dist_hi_optimize[i, j] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
            case _:
                raise

        print("Время работы алгоритма для 3-х линз: ",
              Fore.RED, f"{(time.time() - start_time) // 60} м",
              Fore.GREEN, f"{(time.time() - start_time) % 60} с",
              Fore.WHITE)
        
        # поиск минимума

        # одну линзу перебираем с шагом, а другую сразу весь диапазон
    
        min_idx_optimize_h_i = np.unravel_index(np.argmin(foc_dist_hi_optimize), foc_dist_hi_optimize.shape)
        min_foc_dist_optimize_h_i = foc_dist_hi_optimize[min_idx_optimize_h_i]
        
        fig1, ax1 = self.draw_grid(foc_dist_hi_optimize, 
                                   min_idx_optimize_h_i,
                                   h_range,
                                   min_foc_dist_optimize_h_i,
                                   unit_f_dist,
                                   xlabel_i, ylabel_i, title)

        self._fig_ax = (fig1, ax1)

        plt.show(block=True)
        plt.close()


    def draw_grid(self, 
                  matrix_hi_hj : np.ndarray,
                  idx_h1_h2 : tuple[int, int],
                  h_range : np.ndarray,
                  min_foc_dist : float,
                  unit_f_dist : UnitType, 
                  xlabel : str, ylabel: str, title: str) -> tuple[Figure, Axes]:

        ax : Axes = None

        idx_hx, idx_hy = idx_h1_h2[1], idx_h1_h2[0]
        h_x, h_y = h_range[idx_hx], h_range[idx_hy]

        fig, ax = plt.subplots(1, 1)
        mesh = ax.pcolormesh(h_range,
                             h_range,
                             np.array(matrix_hi_hj) * unit_f_dist.value[1],
                             cmap='viridis', shading='auto')
        fig.colorbar(mesh, ax=ax, label='Фок. отрезок, мм')
        ax.scatter(h_range[idx_hx], h_range[idx_hy], color='red', 
                   label=f"Мин фок.отрезок = {round(min_foc_dist * unit_f_dist.value[1], 3)}" + \
                         f"{unit_to_str_desc(unit_f_dist)}")
        ax.annotate(f'({h_x:.2f}, {h_y:.2f})',
                        xy=(h_x, h_y),
                        xytext=(-15, -15),
                        ha='center', va='top',
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        return (fig, ax)

        
from dataclasses import dataclass
from typing import Any
import time
from colorama import Fore
import os
from os.path import join
from os import getcwd

import numpy as np
import plotly.graph_objects as go


from optimizers.base import BaseLensOptimizer
from utils.unit import UnitType
from utils.keys import *
from validators.input_validator import InputValidator
from data.dataset_helper import DataSetHelper


@dataclass
class VisializeData():
    xlabel: str
    ylabel: str
    zlabel: str
    title: str


class FourLensOptimizer(BaseLensOptimizer):
    def __init__(self):
        super().__init__()
        self.__foc_dist_hi_optimize = None
        self.__h_range = None

    
    def lmbd_focus_dict(self, dataset, heights = None, lambda_massive = None, return_dict=True):
        if dataset['count_linse'] != 4:
            raise Exception(f"Система должна состоять из 4=х линз, а не {self._dataset['count_linse']}")
        return super().lmbd_focus_dict(dataset, heights, lambda_massive, return_dict)
    

    def visualize_depend_f_lmbd(self, units = None, return_fig_ax=False, blockAndShow=True):
        if self._dataset['count_linse'] != 4:
            raise Exception(f"Система должна состоять из 4=х линз, а не {self._dataset['count_linse']}")
        return super().visualize_depend_f_lmbd(units, return_fig_ax, blockAndShow)
    

    def generate_grid_with_fixed_height(self, 
                                        init_h : dict[UnitType, list[float]],
                                        hbounds : tuple[float, float], 
                                        dataset : dict[DataSetKeys, Any],
                                        h_static : Literal['h1', 'h2', 'h3', 'h4'] = 'h1',
                                        visualize_result : bool = True,
                                        units : dict[UnitKeys, UnitType] = None,
                                        unit_f_dist : UnitType = UnitType.MILLIMETER) -> None:
        
        try:
            InputValidator.validate_input_dataset(dataset)
        except (IndexError, KeyError, ValueError):
            dataset = DataSetHelper.create_default_dataset(count_linse=4)
        except Exception:
            raise

        self.__h_range = np.linspace(hbounds[0], hbounds[1], 50)
        self.__foc_dist_hi_optimize = np.zeros((len(self.__h_range), len(self.__h_range), len(self.__h_range)))

        start_time = time.time()

        xlabel, ylabel, zlabel = None, None, None
        title = None
        h_static_val = None

        match h_static:
            case 'h1':
                h_static_val = list(init_h.values())[0][0]
                xlabel, ylabel, zlabel = "h1, мкм", "h2, мкм", "h3, мкм",
                title = f'h1 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(self.__h_range):
                    for j, h_j in enumerate(self.__h_range):
                        for k, h_k in enumerate(self.__h_range):
                            lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_static_val, h_i, h_j, h_k]})
                            self.__foc_dist_hi_optimize[i, j, k] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
            case 'h2':
                h_static_val = list(init_h.values())[0][1]
                xlabel, ylabel, zlabel = "h1, мкм", "h3, мкм", "h4, мкм",
                title = f'h2 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(self.__h_range):
                    for j, h_j in enumerate(self.__h_range):
                        for k, h_k in enumerate(self.__h_range):
                            lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_i, h_static_val, h_j, h_k]})
                            self.__foc_dist_hi_optimize[i, j, k] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
            case 'h3':
                h_static_val = list(init_h.values())[0][2]
                xlabel, ylabel, zlabel = "h1, мкм", "h2, мкм", "h4, мкм",
                title = f'h3 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(self.__h_range):
                    for j, h_j in enumerate(self.__h_range):
                        for k, h_k in enumerate(self.__h_range):
                            lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_i, h_j, h_static_val, h_k]})
                            self.__foc_dist_hi_optimize[i, j, k] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
            case 'h4':
                h_static_val = list(init_h.values())[0][3]
                xlabel, ylabel, zlabel = "h1, мкм", "h2, мкм", "h3, мкм",
                title = f'h4 fixed = {h_static_val}, мкм'
                for i, h_i in enumerate(self.__h_range):
                    for j, h_j in enumerate(self.__h_range):
                        for k, h_k in enumerate(self.__h_range):
                            lmbd_f_hi_optimize = self.lmbd_focus_dict(dataset, heights={UnitType.MICROMETER : [h_i, h_j, h_k, h_static_val]})
                            self.__foc_dist_hi_optimize[i, j, k] = super().calc_focus_dist_static(lmbd_f_hi_optimize)
            case _:
                raise

        print("Время работы алгоритма для 4-х линз: ",
              Fore.RED, f"{(time.time() - start_time) // 60} м",
              Fore.GREEN, f"{(time.time() - start_time) % 60} с",
              Fore.WHITE)
        
        np.save(join(getcwd(), "data", "test", "four_lens_array.npy"), self.__foc_dist_hi_optimize)
        if visualize_result:
            vis_data = VisializeData(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
            self.visualize(vis_data)

    def visualize(self, vis_data : VisializeData = None) -> None:
        
        assert self.__foc_dist_hi_optimize is not None and self.__h_range is not None,\
            "Сначала нужно вызвать generate_grid_with_fixed_height для генерации данных для визуализации"

        min_idx_optimize_h = np.argmin(self.__foc_dist_hi_optimize)
        min_idx = np.unravel_index(min_idx_optimize_h, self.__foc_dist_hi_optimize.shape)
        H1, H2, H3 = np.meshgrid(self.__h_range, self.__h_range, self.__h_range)

        min_x, min_y, min_z = self.__h_range[min_idx[0]], self.__h_range[min_idx[1]], self.__h_range[min_idx[2]]
        min_foc_dist = self.__foc_dist_hi_optimize[min_idx]

        fig = go.Figure\
        (
            data=go.Volume\
                (
                    x=H1.flatten(),
                    y=H2.flatten(),
                    z=H3.flatten(),
                    value=self.__foc_dist_hi_optimize.flatten(),
                    isomin=self.__foc_dist_hi_optimize.min(),
                    isomax=self.__foc_dist_hi_optimize.max(),
                    opacity=0.05,
                    surface_count=15,
                    colorscale='Viridis'
                )
        )

        fig.add_trace\
            (
                go.Scatter3d\
                    (
                        x=[min_x],
                        y=[min_y],
                        z=[min_z],
                        mode='markers+text',
                        marker=dict(size=6, color='red'),
                        text=[f"MIN<br>h2={min_x:.2f}<br>h3={min_y:.2f}<br>h4={min_z:.2f}<br>f={min_foc_dist:.4f}"],
                        textposition="top center",
                        name="Минимум"
                    )
            )

        fig.update_layout\
            (
                title=vis_data.title if vis_data else "Зависимость фокусного расстояния от высот линз",
                scene=dict\
                    (
                        xaxis_title=vis_data.xlabel if vis_data else 'h2',
                        yaxis_title=vis_data.ylabel if vis_data else 'h3',
                        zaxis_title=vis_data.zlabel if vis_data else 'h4',
                    )
            )
        
        fig.show()
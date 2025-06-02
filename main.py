from validators.path_validator import PathValidator
from optimizers.two_lens import TwoLensOptimizer
from optimizers.three_lens import ThreeLensOptimizer
from optimizers.base import BaseLensOptimizer
from optimizers.four_lens import FourLensOptimizer
from utils.unit import UnitType
from data.datasets import DATA_SET_2, DATA_SET_3
from data.dataset_helper import DataSetHelper, DataSetConfig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

if __name__ == '__main__':

    dataset1 = DataSetHelper.create_dataset(count_linse=1,
                                            focus_0 = {1 : 100})
    dataset2 = DataSetHelper.create_dataset(count_linse=2,
                                            focus_0 = {1 : 200, 2 : 200},
                                            harmonica = {1 : 7, 2 : 7.5})
    dataset3 = DataSetHelper.create_dataset(count_linse=3,
                                            focus_0 = {1 : 300, 2 : 300, 3 : 300})
    dataset4 = DataSetHelper.create_dataset(count_linse=4,
                                            focus_0 = {1 : 400, 2 : 400, 3 : 400, 4 : 400})
    
    # fourLensOptimizer = FourLensOptimizer()
    # fourLensOptimizer.generate_grid_with_fixed_height(init_h={UnitType.MICROMETER : [7., 7., 7., 7.]},
    #                                                   hbounds=(5, 15),
    #                                                   dataset=dataset4,
    #                                                   h_static='h1')
    
    baseLensOptimizer = BaseLensOptimizer()
    one_lens_lfd = baseLensOptimizer.lmbd_focus_dict(dataset=dataset1, return_dict=True)
    (fig1, ax1) = baseLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True, blockAndShow=True)

    twoLensOptimizer = TwoLensOptimizer()
    two_lens_lfd = twoLensOptimizer.lmbd_focus_dict(dataset=dataset2, return_dict=True)
    (fig2, ax2) = twoLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True, blockAndShow=False)

    threeLensOptimizer = ThreeLensOptimizer()
    three_lens_lfd = threeLensOptimizer.lmbd_focus_dict(dataset=dataset3, heights={UnitType.MICROMETER : [7, 13.88, 14.18]}, return_dict=True)
    (fig3, ax3) = threeLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True, blockAndShow=False)

    fourLensOptimizer = FourLensOptimizer()
    four_lens_lfd = fourLensOptimizer.lmbd_focus_dict(dataset=dataset4, heights={UnitType.MICROMETER : [7., 14.18, 13.98, 13.78]}, return_dict=True)
    (fig4, ax4) = fourLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True, blockAndShow=False)


    baseLensOptimizer.merge_axes_static(ax1, ax2, 
                                        [baseLensOptimizer.calc_focus_dist_static(one_lens_lfd),
                                         baseLensOptimizer.calc_focus_dist_static(two_lens_lfd)])

    baseLensOptimizer.merge_axes_static(ax1, ax3, 
                                        [baseLensOptimizer.calc_focus_dist_static(one_lens_lfd),
                                         baseLensOptimizer.calc_focus_dist_static(three_lens_lfd)],
                                         labels=["Мин фок.отрезок 1-й линзы", "Мин фок.отрезок 3-х линз"])
    
    baseLensOptimizer.merge_axes_static(ax1, ax4, 
                                        [baseLensOptimizer.calc_focus_dist_static(one_lens_lfd),
                                        0.003807],
                                         labels=["Мин фок.отрезок 1-й линзы", "Мин фок.отрезок 4-х линз"])
    
    # twoLensOptimizer.generate_grid_with_fixed_height(init_h=[7., 7.],
    #                                                   hbounds=(5, 12),
    #                                                   m1=7, m2=7,
    #                                                   h1_or_h2_optimize='h2',
    #                                                   check_neighbour_min=True)
    
    
    #baseLensOptimizer.visualize_dummmy_loss(dataset3)

    # lens1_lfd = baseLensOptimizer.lmbd_focus_dict(dataset1, return_dict=True)
    # (fig1, ax1) = baseLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True)
    
    # lens2_lfd = baseLensOptimizer.lmbd_focus_dict(dataset2, return_dict=True)
    # (fig2, ax2) = baseLensOptimizer.visualize_depend_f_lmbd(return_fig_ax=True)

    # foc_dists = [baseLensOptimizer.calc_focus_dist_static(lens1_lfd) * 1e3, 
    #              baseLensOptimizer.calc_focus_dist_static(lens2_lfd) * 1e3]
    # baseLensOptimizer.merge_axes_static(ax1, ax2, foc_dists)
    
    threeLensOptimizer = ThreeLensOptimizer()

    h_statics = ['h1', 'h2', 'h3']
    for h_static in h_statics:
        threeLensOptimizer.generate_grid_with_fixed_height(init_h={UnitType.MICROMETER : [7., 7., 7.]},
                                                        hbounds=(5, 20),
                                                        dataset=dataset3,
                                                        h_static=h_static)


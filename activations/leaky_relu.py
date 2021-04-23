#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
# author: chenzhen

import copy
import numpy as np

"""
Leaky ReLU Activation
"""


class LeakyReLu():
    """
    This class models an artificial neuraon with leaky ReLU activation function.
    """

    def __init__(self, alpha):
        """
        Initialize weights based on input arguments
        :param alpha:
        :return:
        """
        self.alpha = alpha

    def activation(self, values):
        """

        :param values:
        :return:
        """
        return np.where(values >= 0.0, values, self.alpha * values)

    def derivate(self, values):
        """

        :param values:
        :return:
        """
        return np.where(values >= 0.0, 1, self.alpha)


def plot_xy(fig, pos):
    ax = axisartist.Subplot(fig, pos)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    # ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    # 给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size=1.0)
    # 添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # 设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")


def test():
    """

    :return:
    """
    x = np.linspace(-10, 10, 1000)

    relu = LeakyReLu(0.02)

    forward_result = relu.activation(x)
    backward_result = relu.derivate(x)

    # Plot
    fig = plt.figure(figsize=(16, 6))
    # fig = plt.figure(figsize=(8, 6))
    plot_xy(fig, 121)
    plt.plot(x, forward_result)
    plt.legend(["leaky relu"])

    # fig = plt.figure(figsize=(8, 6))
    plot_xy(fig, 122)
    plt.plot(x, backward_result)
    plt.legend(["derivate leaky relu"])

    plt.show()


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pylab as plt
    import mpl_toolkits.axisartist as axisartist

    mpl.use("TkAgg")


    test()
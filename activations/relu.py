#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
# author: chenzhen

import copy
import numpy as np

"""
ReLU Activation
"""


class ReLU:
    """
    This class models an artificial neuraon with relu activation function.
    """

    def __int__(self):
        """
        Initialize weights based on input arguments
        :return:
        """
        pass

    def activation(self, values):
        """

        :param values:
        :return:
        """
        result = np.maximum(values, 0)
        return result

    def derivate(self, values):
        """

        :param values:
        :return:
        """
        result = copy.deepcopy(values)
        result[result <= 0.0] = 0
        result[result > 0.0] = 1
        return result


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

    relu = ReLU()

    forward_result = relu.activation(x)
    backward_result = relu.derivate(x)

    # Plot
    fig = plt.figure(figsize=(16, 6))
    # fig = plt.figure(figsize=(8, 6))
    plot_xy(fig, 121)
    plt.plot(x, forward_result)
    plt.legend(["relu"])

    # fig = plt.figure(figsize=(8, 6))
    plot_xy(fig, 122)
    plt.plot(x, backward_result)
    plt.legend(["derivate relu"])

    plt.show()


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pylab as plt
    import mpl_toolkits.axisartist as axisartist

    mpl.use("TkAgg")


    test()
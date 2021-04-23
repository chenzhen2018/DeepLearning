#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
# author: chenzhen

import copy
import numpy as np

"""
Some activations
"""


class Sigmoid():
    """Sigmoid Activation Function"""

    def __init__(self):
        """
        Initialize weights based on input arguments
        :return:
        """
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        result = 1 / (1 + np.exp(-x))
        return result

    def gradient(self, x):
        """

        :param values:
        :return:
        """
        result = self.__call__(x) * (1 - self.__call__(x))
        return result


class Tanh():
    """Tanh Activation Function"""

    def __init__(self):
        """
        Initialize weights based on input arguments
        :return:
        """
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return result

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        result = 1 - (self.__call__(x))**2
        return result


class ReLU():
    """ReLU Activation Function"""

    def __init__(self):
        """
        Initialize weights based on input arguments
        :return:
        """
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        result = np.maximum(x, 0)
        return result

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        result = copy.deepcopy(x)
        result[result <= 0.0] = 0
        result[result > 0.0] = 1
        return result


class LeakyReLu():
    """Leaky ReLu Activation Function"""

    def __init__(self, alpha=0.02):
        """
        Initialize weights based on input arguments
        :param alpha:
        :return:
        """
        self.alpha = alpha

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return np.where(x >= 0.0, x, self.alpha * x)

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        return np.where(x >= 0.0, 1, self.alpha)


class ELU():
    """Exponential Linear Units(ELU) Activation Function"""

    def __init__(self, alpha=0.2):
        """
        Initialize weights based on input arguments
        :param alpha:
        :return:
        """
        self.alpha = alpha

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        return np.where(x >= 0.0, 1, np.exp(x) * self.alpha)


class Mish():
    """Mish activations functions"""

    def __init__(self):
        """"""
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        # same expression
        # result = x * ((1 + np.exp(x))**2 - 1) / ((1 + np.exp(x))**2 + 1)
        return x * np.tanh(np.log(1+np.exp(x)))

    # TODO: 暂时不求导
    def gradient(self, x):
        """

        :param x:
        :return:
        """
        pass


class Exp():
    """Exponential activations"""
    def __init__(self):
        pass

    def __call__(self, x):
        return np.exp(x)

    def gradient(self, x):

        return np.exp(x)


class Softsign():
    """Softsign activation function"""
    def __init__(self):
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return x / (np.abs(x) + 1)

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        return 1 / (1 + np.abs(x))**2


class Softplus():
    """Softplus activation function"""
    def __init__(self):
        """

        """
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        return np.exp(x) / (1 + np.exp(x))


class Swith():
    """swith activation function"""
    def __init__(self):
        """

        """
        pass

    def __call__(self, x):
        """

        :param x:
        :return:
        """
        return x / (1 + np.exp(-x))

    def gradient(self, x):
        """

        :param x:
        :return:
        """
        return (1 - np.exp(-x) + x * np.exp(-x)) / (1 + np.exp(-x))**2


dict_activations = {
    'Sigmoid': Sigmoid,
    'Tanh': Tanh,
    'ReLU': ReLU,
    'LeakyReLu': LeakyReLu,
    'ELU': ELU,
    "Mish": Mish,
    "Exp": Exp,
    "Softsign": Softsign,
    "Softplus": Softplus,
    "Swith": Swith
}


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


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pylab as plt
    import mpl_toolkits.axisartist as axisartist

    mpl.use("TkAgg")

    # Get data
    x = np.linspace(-10, 10, 100)

    # TODO: ['Sigmoid', "Tanh", "ReLU", "LeakyReLu", "ELU", "Mish", "Exp", "Softsign", "Softplus", "Swith"]
    # 注意使用LeakReLU和ELU时，可以自定义参数
    activation_name = "Swith"
    activation = dict_activations[activation_name]()

    # Froward & Backward
    forward_result = activation(x)
    backward_result = activation.gradient(x)

    # Plot
    fig = plt.figure(figsize=(16, 6))
    plot_xy(fig, 121)
    plt.plot(x, forward_result)
    plt.legend(["{}".format(activation_name)])

    plot_xy(fig, 122)
    plt.plot(x, backward_result)
    plt.legend(["derivate {}".format(activation_name)])

    plt.show()
    plt.close()
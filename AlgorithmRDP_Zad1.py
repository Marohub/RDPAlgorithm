import math
import numpy as np
from rdp import rdp
import sys

import matplotlib.pyplot as plt


def draw_plots(rdp_points):
    x_axis_RDP, y_axis_RDP = zip(*rdp_points)
    plt.plot(x_axis_RDP, y_axis_RDP)
    plt.show()


def function(x):
    return math.sin(3*math.pi*x)*x


def get_points():
    points = []
    for x in np.arange(0.0, 5.0, 0.01):
        points.append([x, function(x)])
    return points


def main():
    points = np.array(get_points())
    draw_plots(points)
    return 0


class RDP:

    def add_to_rdp_array(self, index):
        x, y = self.get_point(index)
        self.rdp_array.append([x, y])



    def run(self):
        self.add_to_rdp_array(0)
        self.algorithm(0, len(self.points)-1)
        self.add_to_rdp_array(len(self.points)-1)
        return np.array(self.rdp_array)


if __name__ == "__main__":
    sys.exit(main())
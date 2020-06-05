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
    epsilon = 1.0
    custom_rdp = RDP(points, epsilon).run()
    draw_plots(custom_rdp)
    return 0


class RDP:

    def __init__(self, points, epsilon):
        self.rdp_array = []
        self.points = points
        self.epsilon = epsilon

    def get_point(self, index):
        return self.points[index][0], self.points[index][1]

    def perpendicular_distance(self, A, B, C, index):
        x, y = self.get_point(index)
        numerator = abs(A*x+B*y+C)
        denominator = math.sqrt(A**2 + B**2)
        return numerator/denominator

    def get_line_coefficients(self, start_index, end_index):
        # Ax + By + C = 0
        A = C = 0
        B = -1
        x1, y1 = self.get_point(start_index)
        x2, y2 = self.get_point(end_index)
        if x1 == x2:
            A = x2
        else:
            A = (y2 - y1)/(x2 - x1)
            C = y1 - A*x1
        return A, B, C

    def find_furthest(self, start_index, end_index):
        A, B, C = self.get_line_coefficients(start_index, end_index)
        index = dmax = -1
        for i in range(start_index, end_index):
            d = self.perpendicular_distance(A, B, C, i)
            if d > dmax:
                index = i
                dmax = d
        if dmax > self.epsilon:
            return index
        else:
            return -1

    def add_to_rdp_array(self, index):
        x, y = self.get_point(index)
        self.rdp_array.append([x, y])

    def algorithm(self, start_index, end_index):
        next_index = self.find_furthest(start_index, end_index)

    def run(self):
        self.add_to_rdp_array(0)
        self.algorithm(0, len(self.points)-1)
        self.add_to_rdp_array(len(self.points)-1)
        return np.array(self.rdp_array)


if __name__ == "__main__":
    sys.exit(main())
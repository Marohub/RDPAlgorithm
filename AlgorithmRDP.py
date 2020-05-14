import math
import numpy as np
from rdp import rdp
import sys

import matplotlib.pyplot as plt

def draw_plots(rdp_points, original_points):
    x_axis_RDP, y_axis_RDP = zip(*rdp_points)
    x_axis_original, y_axis_original = zip(*original_points)
    plt.plot(x_axis_RDP, y_axis_RDP)
    plt.plot(x_axis_original, y_axis_original)
    plt.show()

def main():
    points = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1], [3.0, 5.0], [4.0, 6.0], [5.0, 7.1], [6.0, 8.1], [7.0, 9.0], [8.0, 9.0], [9.0, 9.0]])
    epsilon = 1.0
    x = rdp(points, epsilon)
    y = RDP().generalize(epsilon, points)
    np.testing.assert_array_equal(x, y)
    draw_plots(y, points)
    return 0


class RDP:
    def __init__(self):
        self.A = 0
        self.B = -1
        self.C = 0

    def algorithm(self, epsilon, points) :
        dmax = 0
        index = 0
        end = len(points) - 1
        self.make_line(points)
        for i in range(1, end):
            d = self.perpendicular_distance(i, points)
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            points_1 = points[0:index+1]
            points_2 = points[index:end+1]
            rec_results_1 = self.algorithm(epsilon, points_1)
            rec_results_2 = self.algorithm(epsilon, points_2)
        else:
            rec_results_1 = [points[0]]
            rec_results_2 = [points[end]]

        result_points = np.concatenate((rec_results_1, rec_results_2), axis = 0)
        return result_points

    def perpendicular_distance(self, k, points):
        x = (k, 0)
        y = (k, 1)
        numerator = abs(self.A*points[x]+self.B*points[y]+self.C)
        denominator = math.sqrt(self.A**2 + self.B**2)
        return numerator/denominator

    def make_line(self, points):
        end = len(points) - 1
        x1 = (0, 0)
        y1 = (0, 1)
        x2 = (end, 0)
        y2 = (end, 1)
        if (points[x2] - points[x1]) is 0:
            self.B = 0
            self.A = points[x2]
            self.C = 0
        else :
            self.A = (points[y2] - points[y1])/(points[x2] - points[x1])
            self.C = points[y1] - self.A*points[x1]

    def generalize(self, epsilon, points):
        result_points = self.algorithm(epsilon, points)
        result_points = self.reduce(result_points)
        return result_points


    def reduce(self, result_points):
        end = len(result_points)-1
        i = 2
        while i < end:
            result_points = np.delete(result_points, i, 0)
            i += 1
            end = len(result_points)-1
        return result_points


if __name__ == "__main__":
    sys.exit(main())

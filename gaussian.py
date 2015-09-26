__author__ = 'mingles'

import numpy as np
import numpy.linalg as lin
import math
import scipy

from scipy.stats import *

class Classify():

    def __init__(self, training_set, training_classes):

        self.training_set = training_set
        self.training_classes = training_classes
        self.training_set_by_class = self.organise_by_class()
        self.means = self.get_means()
        self.cov_matrices = self.get_cov_matrices()
        self.prior_probability = 0.1

    def organise_by_class(self):
        training_set_by_class = []
        for x in range(1, 11):
            xarray = []
            for i in range(0, len(self.training_classes)):
                if self.training_classes[i] == [x]:
                    xarray.append(self.training_set[i])
            training_set_by_class.append(xarray)
        return training_set_by_class

    def get_means(self):
        means = []
        for x in self.training_set_by_class:
            average = np.matrix(x).mean(0).tolist()[0]
            if average == []:
                average = [0, 0, 0]
            means.append(average)
        return means

    def get_cov_matrices(self):
        cov_matricies = []
        for x in self.training_set_by_class:
            cov_matricies.append(np.cov(np.array(x).T))
        return cov_matricies

    def gaussian(self):
        # for i in range(len(self.means)):
        return 5



    def get_class(self, test_point):
        probabilities = []
        for i in range(0,10):#len(self.means)):

            x = np.array(test_point)
            mu = np.array(self.means[i])
            d = len(test_point)
            cov = self.cov_matrices[i]

            cov = cov.tolist()
            for i in range(len(cov)):
                for j in range(len(cov[i])):
                    if i != j:
                        cov[i][j] = 0

            cov = np.matrix(cov)
            det = lin.det(cov)

            # part1 = -(x.size/2)*np.log(2*math.pi) - (x.size/2)*np.log(lin.det(cov))
            # part2 = 0.5 * ((x-mu) * (x-mu).T) * lin.inv(cov)
            # print part1 - part2

            k = x.shape[0]
            part1 = np.exp(-0.5*k*np.log(2*np.pi))
            part2 = (det ** -0.5)
            dev = x-mu
            part3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(cov)), dev))
            dmvnorm = (part1*part2*part3)

            # eq1 = 1/(2 * math.pi ** (d / 2))
            # eq2 = det ** -0.5
            # eq3 = np.exp((- 0.5 * (x - mu).T * lin.inv(cov) * (x - mu)))
            # print eq3
            probabilities.append(np.matrix(dmvnorm).tolist()[0][0])

        over = 0
        for i in range(len(probabilities)):
            over = over + (0.1 * probabilities[i])

        if over == 0:
            over = 0.00000000000000000001

        bayes = []
        for i in range(len(probabilities)):
            bayes.append(probabilities[i] * 0.1 / over)

        max_prob = bayes[0]
        max_index = 0
        for i in range(len(bayes)):
            if bayes[i] > max_prob:
                max_prob = bayes[i]
                max_index = i

        return max_index + 1


def main():

    training_set = [[1, 2, 3], [1, 3, 2], [1, 1, 2], [5, 6, 7], [6, 6, 6], [10, 11, 12], [11, 10, 11]]
    training_classes = [[1], [1], [1], [2], [2], [3], [3]]
    test_point = [1, 2, 2]


    c = Classify(training_set, training_classes)
    # c.get_class(test_point)


if __name__ == '__main__':
    main()
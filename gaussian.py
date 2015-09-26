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
            det = lin.det(2**19*cov)
            det = 1
            # normals = []
            # for p in (self.cov_matrices):
            #     newdet = lin.det(2**19*p)
            #     print newdet
            #     normals.append(newdet)
            #
            # normals = norm(normals)

            # print (lin.det(self.cov_matrices[i]*10000000))
            # print det
            k = x.shape[0]
            part1 = np.exp(-0.5*k*np.log(2*np.pi))
            part2 = (det ** -0.5)
            dev = x-mu
            # print -0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(cov)), dev)/10000
            part3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(cov)), dev))
            dmvnorm = part1*part2*part3

            print dmvnorm


            # eq1 = 1/(2 * math.pi ** (d / 2))
            # eq2 = det ** -0.5
            # eq3 = np.exp((- 0.5 * (x - mu).T * lin.inv(cov) * (x - mu)))
            # print eq3
            probabilities.append(x)

        return probabilities


def main():

    training_set = [[1, 2, 3], [1, 3, 2], [1, 1, 2], [5, 6, 7], [6, 6, 6], [10, 11, 12], [11, 10, 11]]
    training_classes = [[1], [1], [1], [2], [2], [3], [3]]
    test_point = [1, 2, 2]


    c = Classify(training_set, training_classes)
    # c.get_class(test_point)


if __name__ == '__main__':
    main()
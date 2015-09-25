__author__ = 'mingles'

import numpy as np


class Classify():

    def __init__(self, k, test_val, training_set, training_classes):
        self.k = k
        self.test_val = test_val
        self.training_set = training_set
        self.training_classes = training_classes
        self.nearest_neighbour_class = [[-1]]
        self.distance_values = []
        self.nearest_classes = []

    def get_nn(self):

        # find distances between test value and all points in training set
        for y in range(0, len(self.training_set)):
            training_val = self.training_set[y]
            distance = np.linalg.norm(np.array(abs(self.test_val - training_val)))
            self.distance_values.append((y, distance))

        # sort get classes by closest distance values and pick first k
        data_type = [('position', int), ('difference', object)]
        self.distance_values = np.sort(np.array(self.distance_values, dtype=data_type), order='difference')
        for distance in self.distance_values[:self.k]['position']:
            self.nearest_classes.append(self.training_classes[distance])

        # return mode of distance values
        return self.knn_mode()

    # Finds the mode. If >1 classes have the greatest number of points in the
    # k nearest neighbours, the furthest point is removed until the tie is broken.
    def knn_mode(self):

        # count occurrences of classes
        done = []
        the_result = []
        for a in self.nearest_classes:
            count = 0
            for b in self.nearest_classes:
                if a not in done:
                    if a == b:
                        count += 1
            done.append(a)
            the_result.append((a, count))

        # find most common occurrence and break ties
        max_occurrence = 0
        mode = -1
        for c in the_result:
            if c[1] > max_occurrence:
                max_occurrence = c[1]
                mode = c[0]
            elif c[1] == max_occurrence:
                # remove furthest point and repeat function
                self.nearest_classes = self.nearest_classes[:(len(self.nearest_classes) - 1)]
                self.knn_mode()
        return mode
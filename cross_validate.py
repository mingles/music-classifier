import numpy as np
from numpy import *
import scipy.io

__author__ = 'mingles'


def main():

    # loading data into seperate matrices
    data = scipy.io.loadmat("gtzan.mat")
    features = []
    classes = []

    for i in (range(1, 11)):
        features.append(matrix(data['fold%d_features' % i]))
        classes.append(matrix(data['fold%d_classes' % i]))

    correctly_classified = 0
    total = 0
    for folds in features:
        total += (len(folds))
    print "total " + str(total)

    # iterating through first fold
    for i in range(len(features)):
        correctly_classified_fold = 0
        test_set = features[i]
        test_classes = classes[i]

        training_set = []
        training_classes = []
        training_check = []

        for j in range(len(features)):
            if i != j:
                for k in range(len(features[j])):
                    training_set.append(features[j][k])
                    training_classes.append(classes[j][k])

                training_check.append(j)

        for x in range(len(test_set)):
            difference_values = []
            test_val = test_set[x]
            test_class = test_classes[x]

            for y in range(0, len(training_set)):
                training_val = training_set[y]
                difference = np.linalg.norm(np.array(abs(test_val - training_val)))
                # print y
                # print difference
                difference_values.append((y, difference))

            data_type = [('position', int), ('difference', object)]
            results = np.array(difference_values, dtype=data_type)
            results = np.sort(results, order='difference')
            nearest_classes = []
            for result in results[:1]['position']:
                nearest_classes.append(training_classes[result])

            # get mode and compare assigned class to actual class
            if knn_mode(nearest_classes) == test_class:
                correctly_classified_fold += 1

        print "Testing Fold " + str(i + 1) + ": " + str(correctly_classified_fold) + "/" + str(len(test_classes))
        correctly_classified = correctly_classified + correctly_classified_fold

    print "Combined Result: " + str(correctly_classified) + "/" + str(total)
    print "Percentage Correctly Classified: " + str(("{0:.2f}".format(float(correctly_classified) / float(total))))


def knn_mode(nearest_classes):
    done = []
    the_result = []
    for a in nearest_classes:
        count = 0
        for b in nearest_classes:
            if a not in done:
                if a == b:
                    count += 1
        done.append(a)
        the_result.append((a, count))

    max_occurrence = 0
    mode = -1
    for c in the_result:
        if c[1] > max_occurrence:
            max_occurrence = c[1]
            mode = c[0]
        elif c[1] == max_occurrence:
            # remove furthest element and repeat function
            nearest_classes = nearest_classes[:(len(nearest_classes) - 1)]
            knn_mode(nearest_classes)
    return mode

if __name__ == '__main__':
    main()

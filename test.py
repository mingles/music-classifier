__author__ = 'mingles'

import numpy as np

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

    max_occurance = 0
    mode = -1
    for c in the_result:
        if c[1] > max_occurance:
            max_occurance = c[1]
            mode = c[0]
        elif c[1] == max_occurance:
            # "remove end and redo"
            nearest_classes = nearest_classes[:(len(nearest_classes) - 1)]
            knn_mode(nearest_classes)
    return mode

def main():
    grades = [62, 73, 70, 64, 56, 72, 67, 58, 74, 64, 72]
    print np.median(grades)
    print np.average(grades)
    print np.std(grades)
if __name__ == '__main__':
    main()
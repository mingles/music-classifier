__author__ = 'mingles'
import scipy.io
import knn


class CrossValidate():

    def __init__(self, classifier):
        print "Cross Validating with Classifier " + str(classifier)
        data = scipy.io.loadmat("gtzan.mat")
        features = []
        classes = []
        classifier = classifier

        for i in (range(1, 11)):
            features.append(data['fold%d_features' % i])
            classes.append(data['fold%d_classes' % i])

        correctly_classified = 0
        total = 0
        for folds in features:
            total += (len(folds))

        # iterating through each fold
        for i in range(len(features)):
            correctly_classified_fold = 0
            test_set = features[i]
            test_classes = classes[i]

            training_set = []
            training_classes = []

            # creating training set for fold i
            for j in range(len(features)):
                if i != j:
                    for k in range(len(features[j])):
                        training_set.append(features[j][k])
                        training_classes.append(classes[j][k])

            # classifying the test_set against the training_set
            for x in range(len(test_set)):
                test_val = test_set[x]
                test_class = test_classes[x]
                classified_class = [[-1]]

                # create classifier
                if len(classifier) > 1 and classifier[0] == "knn":
                    k = knn.Knn(classifier[1], test_val, training_set, training_classes)
                    classified_class = k.get_nn()
                else:
                    print "Invalid Classifier Input"
                    quit()

                # checking if test_data was classified correctly
                if classified_class == test_class:
                    correctly_classified_fold += 1

            print "Testing Fold " + str(i + 1) + ": " + str(correctly_classified_fold) + "/" + str(len(test_classes))
            correctly_classified = correctly_classified + correctly_classified_fold

        print "Combined Result: " + str(correctly_classified) + "/" + str(total)
        print "Percentage Correctly Classified: " + str(("{0:.2f}".format(float(correctly_classified) / float(total))))


def main():
    CrossValidate(("knn", 1))

if __name__ == '__main__':
    main()
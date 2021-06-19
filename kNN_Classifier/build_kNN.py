#!/usr/bin/env python3

# ###################################################################################################
# 
# David Roesler (roeslerdavidthomas@gmail.com) "build_kNN.py"
#
# ###################################################################################################
#
# kNN Classifier
#
# This script builds a kNN model from the training data, classifies the training and test data, and calculates
# the accuracy.
#
# The format for launching the script is:
# build_kNN.sh training_data test_data k_val similarity_func sys_output > acc_file
#
# ###################################################################################################

import sys      # read from std in
import re       # used to process input files
import time
import math     # used for log function
import numpy as np


class KNN:
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    k_val = int(sys.argv[3])
    similarity_func = int(sys.argv[4])
    sys_output_filename = sys.argv[5]

    #  ===========================================================================================
    #  Default constructor.  Initializes data members.
    #  ===========================================================================================

    def __init__(self):
        self.train_vector_list = []     # [train instance id] = [[feat1,count],[feat2,count],...]
        self.test_vector_list = []      # [test instance id] = [[feat1,count],[feat2,count],...]
        self.train_vector_set_list = []  # [train instance id] = {feat_id1, feat_id2, ...}
        self.test_vector_set_list = []   # [test instance id] = {feat_id1, feat_id2, ...}

        self.class_set = set()           # set of training classes
        self.class_list = []             # indexed list of training class labels [class id] = class name
        self.class_to_id = {}            # [class name] = class id (index in class_list)

        self.test_class_set = set()      # set of test classes not found in training data
        self.test_class_list = []        # indexed list of training class labels followed by new test class labels
        self.test_class_to_id = {}       # [test class name] = id number (index in test_class_list)

        self.feat_set = set()
        self.feat_to_id = {}             # [feature name] = id number (index in feat_list)
        self.feat_list = []              # indexed list of features [feature id] = feature name

        self.inst_feat_matrix = []       # sparse matrix [train_vector_list index (instance id)][feature id] = count
        self.test_inst_feat_matrix = []  # sparse matrix [test_vector_list index (instance id)][feature id] = count

        self.inst_feat_square_matrix = []
        # sparse matrix [train_vector_list index (instance id)][feature id] = count squared
        self.test_inst_feat_square_matrix = []
        # sparse matrix [test_vector_list index (instance id)][feature id] = count squared

        self.train_dist_matrix = []     # [train_id][train_id] = distance or cosine similarity (float)
        self.test_dist_matrix = []      # [test_id][train_id] = distance or cosine similarity (float)

        self.train_vector_lengths = []    # [train inst id] = vector length
        self.test_vector_lengths = []     # [test inst id] = vector length

        self.train_confusion_matrix = []  # [true training class id][system training class id] = training instance count
        self.test_confusion_matrix = []   # [true test class id][system test class id] = test instance count

        self.num_of_classes = 0
        self.num_of_feats = 0
        self.num_of_instances = 0
        self.num_of_test_instances = 0

        self.sys_output_train_list = []
        # list of training strings to write to sys_output file. index matches instance id.
        self.sys_output_test_list = []
        # list of test strings to write to sys_output file. index matches instance id.
        self.temp_sys_out_result_list = []  # temporary result list for storing tuples (class id, prob)

    # =====================================================================
    # read_train_data:
    # Called from main. Reads instances and features
    # from training file data.
    # =====================================================================

    def read_train_data(self):
        with open(KNN.train_filename, mode="r") as input_file:  # open training file
            content_list = input_file.read().splitlines(False)
            line_index = 0
            for line in content_list:
                if line:
                    word_list = line.split()            # split on whitespace
                    self.train_vector_list.append([word_list[0]])      # class name is first item in list
                    if word_list[0] not in self.class_set:             # check if class label has been seen
                        self.class_list.append(word_list[0])
                        self.class_set.add(word_list[0])
                    for i in range(1, len(word_list)):
                        word_match = re.match(r'^([^:]+):([0-9]+)', word_list[i])
                        if word_match:
                            self.train_vector_list[line_index].append([word_match.group(1), int(word_match.group(2))])
                            if word_match.group(1) not in self.feat_set:   # check if feature has been seen
                                self.feat_list.append(word_match.group(1))
                            self.feat_set.add(word_match.group(1))
                    self.train_vector_list[line_index][1:] = \
                        sorted(self.train_vector_list[line_index][1:], key=lambda x: x[0])
                line_index += 1
        input_file.close()

    # =====================================================================
    # read_test_data:
    # Called from main. Reads instances and features
    # from test file data.
    # =====================================================================

    def read_test_data(self):
        self.class_list.sort()                                     # sort training class list alphabetically
        self.feat_list.sort()                                      # sort feat list alphabetically
        self.test_class_list = self.class_list.copy()             # copy training classes to test class list
        with open(self.test_filename, mode="r") as input_file:     # open training file
            content_list = input_file.read().splitlines(False)
            line_index = 0
            for line in content_list:
                if line:
                    word_list = line.split()            # split on whitespace
                    if word_list[0] not in self.class_set and word_list[0] not in self.test_class_set:
                        self.test_class_list.append(word_list[0])      # check if class label has been seen
                        self.test_class_set.add(word_list[0])
                    self.test_vector_list.append([word_list[0]])       # class name is first item in list
                    for i in range(1, len(word_list)):
                        word_match = re.match(r'^([^:]+):([0-9]+)', word_list[i])
                        if word_match:
                            if word_match.group(1) in self.feat_set:
                                self.test_vector_list[line_index].append([word_match.group(1),
                                                                          int(word_match.group(2))])
                                # only retain features that are in training data
                    self.test_vector_list[line_index][1:] = \
                        sorted(self.test_vector_list[line_index][1:], key=lambda x: x[0])
                line_index += 1
        input_file.close()

    # =====================================================================
    # prep_data_structures:
    # Called from main. Initializes and fills data structures that
    # facilitate operations.
    # =====================================================================

    def prep_data_structures(self):
        self.num_of_classes = len(self.class_list)
        self.num_of_feats = len(self.feat_list)
        self.num_of_instances = len(self.train_vector_list)
        self.num_of_test_instances = len(self.test_vector_list)

        for i in range(0, self.num_of_feats):
            self.feat_to_id[self.feat_list[i]] = i    # fill feat_to_id dictionary
        for i in range(0, self.num_of_classes):
            self.class_to_id[self.class_list[i]] = i    # fill class_to_id dictionary
        for i in range(0, len(self.test_class_list)):
            self.test_class_to_id[self.test_class_list[i]] = i  # fill test_class_to_id dictionary
        for i in range(0, self.num_of_instances):    # replace class names with ids in training instance list
            self.train_vector_list[i][0] = self.class_to_id[self.train_vector_list[i][0]]
        for i in range(0, len(self.test_vector_list)):    # replace class names with ids in test instance list
            self.test_vector_list[i][0] = self.test_class_to_id[self.test_vector_list[i][0]]

        self.sys_output_train_list = [None] * self.num_of_instances         # initialize to size of train instance list
        self.sys_output_test_list = [None] * self.num_of_test_instances     # initialize to size of test instance list
        self.temp_sys_out_result_list = [(0, 0.0)] * self.num_of_classes    # list of tuples (class id, prob)

        self.train_vector_set_list = [set()] * self.num_of_instances        # initialize to size of train instance list
        self.test_vector_set_list = [set()] * self.num_of_test_instances    # initialize to size of test instance list

        # initialize confusion matrices
        self.train_confusion_matrix = \
            [[0 for column in range(0, self.num_of_classes)] for row in range(0, self.num_of_classes)]
        self.test_confusion_matrix = \
            [[0 for column in range(0, len(self.test_class_list))] for row in range(0, len(self.test_class_list))]

        # initialize instance (row) x feature (column) matrix values to 0
        self.inst_feat_matrix = \
            [[0 for column in range(0, self.num_of_feats)] for row in range(0, self.num_of_instances)]
        self.inst_feat_square_matrix = \
            [[0 for column in range(0, self.num_of_feats)] for row in range(0, self.num_of_instances)]

        # initialize instance (row) x feature (column) matrix values to 0
        self.test_inst_feat_matrix = \
            [[0 for column in range(0, self.num_of_feats)] for row in range(0, self.num_of_test_instances)]
        self.test_inst_feat_square_matrix = \
            [[0 for column in range(0, self.num_of_feats)] for row in range(0, self.num_of_test_instances)]

        for inst_index in range(0, self.num_of_instances):
            for feat_index in range(1, len(self.train_vector_list[inst_index])):
                feat_id_value = self.feat_to_id[self.train_vector_list[inst_index][feat_index][0]]
                # fill the matrix with count values for features that appear in the instance vectors
                self.inst_feat_matrix[inst_index][feat_id_value] = self.train_vector_list[inst_index][feat_index][1]
                # fill the square matrix with squared count values
                self.inst_feat_square_matrix[inst_index][feat_id_value] = \
                    self.train_vector_list[inst_index][feat_index][1] ** 2
                # replace feat strings with feat ids in original training vector list
                self.train_vector_list[inst_index][feat_index] = feat_id_value
            self.train_vector_set_list[inst_index] = set(self.train_vector_list[inst_index][1:])  # SET VERSION

        for inst_index in range(0, self.num_of_test_instances):
            for feat_index in range(1, len(self.test_vector_list[inst_index])):
                feat_id_value = self.feat_to_id[self.test_vector_list[inst_index][feat_index][0]]
                # fill the matrix with count values for features that appear in the instance vectors
                self.test_inst_feat_matrix[inst_index][feat_id_value] = self.test_vector_list[inst_index][feat_index][1]
                # fill the square matrix with squared count values
                self.test_inst_feat_square_matrix[inst_index][feat_id_value] = \
                    self.test_vector_list[inst_index][feat_index][1] ** 2
                # replace feat strings with feat ids in original test vector list
                self.test_vector_list[inst_index][feat_index] = feat_id_value
            self.test_vector_set_list[inst_index] = set(self.test_vector_list[inst_index][1:])  # SET VERSION

    # =====================================================================
    # calc_vector_lengths:
    # Called by run_test. Calculates vector lengths for training and
    # test vectors.
    # =====================================================================

    def calc_vector_lengths(self):
        # calc test vector lengths
        self.test_vector_lengths = [0.0] * self.num_of_test_instances
        for inst_i_id in range(0, self.num_of_test_instances):
            total = 0
            for feat_id in self.test_vector_set_list[inst_i_id]:
                total += self.test_inst_feat_square_matrix[inst_i_id][feat_id]
            self.test_vector_lengths[inst_i_id] = math.sqrt(total)
        # calc train vector lengths
        self.train_vector_lengths = [0.0] * self.num_of_instances
        for inst_j_id in range(0, self.num_of_instances):
            total = 0
            for feat_id in self.train_vector_set_list[inst_j_id]:
                total += self.inst_feat_square_matrix[inst_j_id][feat_id]
            self.train_vector_lengths[inst_j_id] = math.sqrt(total)

    # =====================================================================
    # run_test:
    # Called from main. Wrapper for run_test function.
    # =====================================================================

    def run_test(self):
        if KNN.similarity_func == 1:                                    # test using euclidean distance (1)
            # initialize distance matrices to distance of 0.0
            self.train_dist_matrix = \
                [[0.0 for column in range(0, self.num_of_instances)] for row in range(0, self.num_of_instances)]
            self.test_dist_matrix = \
                [[0.0 for column in range(0, self.num_of_instances)] for row in range(0, self.num_of_test_instances)]

            # test training data
            self.distance_test(self.sys_output_train_list, self.train_vector_list, self.class_list,
                               self.train_confusion_matrix, self.train_dist_matrix, self.train_vector_set_list,
                               self.inst_feat_matrix, 1, self.train_vector_lengths,
                               self.num_of_instances)
            # test test data
            self.distance_test(self.sys_output_test_list, self.test_vector_list, self.test_class_list,
                               self.test_confusion_matrix, self.test_dist_matrix, self.test_vector_set_list,
                               self.test_inst_feat_matrix, 0,
                               self.test_vector_lengths, self.num_of_test_instances)

        else:                                                           # test using cosine similarity (2)
            # initialize distance matrices to cosine similarity of 1.0
            self.train_dist_matrix = \
                [[1.0 for column in range(0, self.num_of_instances)] for row in range(0, self.num_of_instances)]
            self.test_dist_matrix = \
                [[1.0 for column in range(0, self.num_of_instances)] for row in range(0, self.num_of_test_instances)]

            # test training data
            self.cosine_test(self.sys_output_train_list, self.train_vector_list, self.class_list,
                             self.train_confusion_matrix, self.train_dist_matrix, 1, self.train_vector_set_list,
                             self.inst_feat_matrix, self.num_of_instances, self.train_vector_lengths)
            # test test data
            self.cosine_test(self.sys_output_test_list, self.test_vector_list, self.test_class_list,
                             self.test_confusion_matrix, self.test_dist_matrix, 0, self.test_vector_set_list,
                             self.test_inst_feat_matrix, self.num_of_test_instances, self.test_vector_lengths)

    # =====================================================================
    # distance_test:
    # Called by run_test. Tests training or test data on training data using
    # distance metric. Arguments determine which data is tested.
    # column_increment_value is 1 for train vs train.
    # column_increment_value is 0 for test vs train.
    # =====================================================================

    def distance_test(self, sys_output_list: list, vector_list: list, class_list: list,
                      confusion_matrix: list, dist_matrix: list, vector_set_list: list,
                      count_matrix: list, column_increment_value: int,
                      vector_lengths: list, i_range: int):
        start_column = 0
        for inst_i_id in range(0, i_range):   # i is row / test
            start_column += column_increment_value    # for train vs. train, only iterate on upper right of matrix
            test_set = vector_set_list[inst_i_id]
            test_matrix_prefix = count_matrix[inst_i_id]
            for inst_j_id in range(start_column, self.num_of_instances):  # j is column / train
                train_set = self.train_vector_set_list[inst_j_id]
                train_matrix_prefix = self.inst_feat_matrix[inst_j_id]
                intersect_set = test_set.intersection(train_set)    # create set of intersection features
                dot_xy = 0.0
                for feat_id in intersect_set:   # for each feature in shared set
                    dot_xy += (train_matrix_prefix[feat_id] * test_matrix_prefix[feat_id])  # sum intersect dot product
                dot_xy = math.sqrt(dot_xy)  # get sqrt to match format of stored vector length values
                dist_sum = math.sqrt((vector_lengths[inst_i_id] ** 2) + (self.train_vector_lengths[inst_j_id] ** 2) -
                                     (2 * (dot_xy ** 2)))
                dist_matrix[inst_i_id][inst_j_id] = dist_sum    # store result
                if column_increment_value:      # for train vs train, also write to other side of symmetrical matrix
                    dist_matrix[inst_j_id][inst_i_id] = dist_sum
            self.find_dist_neighbors(sys_output_list, vector_list, inst_i_id, dist_matrix,
                                     confusion_matrix, class_list)

    # =====================================================================
    # find_dist_neighbors:
    # Called by distance_test. Finds
    # k nearest neighbors based on distance metric.
    # =====================================================================

    def find_dist_neighbors(self, sys_output_list: list, vector_list: list, inst_i_id: int, dist_matrix: list,
                            confusion_matrix: list, class_list: list):

        class_count = [0] * self.num_of_classes
        # make numpy array from results and get list of k lowest values
        nearest_list = np.argpartition(np.array(dist_matrix[inst_i_id]), KNN.k_val)[:KNN.k_val]
        for i in nearest_list:
            class_count[self.train_vector_list[i][0]] += 1  # count classes of nearest neighbors
        for i in range(0, self.num_of_classes):
            self.temp_sys_out_result_list[i] = (i, (class_count[i] / KNN.k_val))
        # sort in descending order based on prob value
        self.temp_sys_out_result_list = sorted(self.temp_sys_out_result_list, key=lambda x: x[1], reverse=True)
        true_class_id = vector_list[inst_i_id][0]
        temp_string = class_list[true_class_id]
        for value_pair in self.temp_sys_out_result_list:
            temp_string += (" " + self.class_list[value_pair[0]] + " " + str(round(value_pair[1], 5)))
        sys_output_list[inst_i_id] = temp_string
        class_id_prediction = self.temp_sys_out_result_list[0][0]
        confusion_matrix[true_class_id][class_id_prediction] += 1  # update confusion matrix

    # =====================================================================
    # cosine_test:
    # Called by run_test. Tests training data on training data using
    # cosine similarity metric.
    # =====================================================================

    def cosine_test(self, sys_output_list: list, vector_list: list, class_list: list,
                    confusion_matrix: list, dist_matrix: list, column_increment_value: int,
                    vector_set_list: list, count_matrix: list, i_range: int, vector_lengths: list):
        start_column = 0
        for inst_i_id in range(0, i_range):
            start_column += column_increment_value
            test_set = vector_set_list[inst_i_id]
            test_matrix_prefix = count_matrix[inst_i_id]
            for inst_j_id in range(start_column, self.num_of_instances):  # only iterate on upper right of matrix
                # i is row (test), j is column (train)
                train_set = self.train_vector_set_list[inst_j_id]
                train_matrix_prefix = self.inst_feat_matrix[inst_j_id]
                intersect_set = test_set.intersection(train_set)
                numerator_sum = 0
                for i in intersect_set:  # for i in shared set
                    numerator_sum += train_matrix_prefix[i] * test_matrix_prefix[i]
                cosine_similarity = \
                    numerator_sum / (vector_lengths[inst_i_id] * self.train_vector_lengths[inst_j_id])

                dist_matrix[inst_i_id][inst_j_id] = cosine_similarity
                if column_increment_value:  # if train vs train, store value in both sides of the symmetrical matrix
                    dist_matrix[inst_j_id][inst_i_id] = cosine_similarity

            self.find_cosine_neighbors(sys_output_list, vector_list, inst_i_id, dist_matrix, confusion_matrix,
                                       class_list)

    # =====================================================================
    # find_cosine_neighbors:
    # Called by distance_train_test and distance_test_test. Finds
    # k nearest neighbors based on distance metric.
    # =====================================================================

    def find_cosine_neighbors(self, sys_output_list: list, vector_list: list, inst_i_id: int, dist_matrix: list,
                              confusion_matrix: list, class_list: list):

        dist_array = np.array(dist_matrix[inst_i_id])  # make numpy array from results
        nearest_list = np.argpartition(dist_array, -KNN.k_val)[-KNN.k_val:]  # get list of k lowest values
        class_count = [0] * self.num_of_classes
        for i in nearest_list:
            class_count[self.train_vector_list[i][0]] += 1  # count classes of nearest neighbors
        for i in range(0, self.num_of_classes):
            self.temp_sys_out_result_list[i] = (i, (class_count[i] / KNN.k_val))
        # sort in descending order based on prob value
        self.temp_sys_out_result_list = sorted(self.temp_sys_out_result_list, key=lambda x: x[1], reverse=True)
        true_class_id = vector_list[inst_i_id][0]
        temp_string = class_list[true_class_id]
        for value_pair in self.temp_sys_out_result_list:
            temp_string += (" " + self.class_list[value_pair[0]] + " " + str(round(value_pair[1], 5)))
        sys_output_list[inst_i_id] = temp_string
        class_id_prediction = self.temp_sys_out_result_list[0][0]
        confusion_matrix[true_class_id][class_id_prediction] += 1   # update confusion matrix

    # =====================================================================
    # print_output_files:
    # Called by main.
    # Prints sys_output file.
    # =====================================================================

    def print_output_files(self):
        with open(KNN.sys_output_filename, mode="w", newline="\n", encoding="utf-8") as sys_output_file:
            sys_output_file.write("%%%%% training data:\n")
            for i in range(0, len(self.sys_output_train_list)):
                sys_output_file.write("array:" + str(i) + " " + self.sys_output_train_list[i] + "\n")
            sys_output_file.write("\n\n%%%%% test data:\n")
            for i in range(0, len(self.sys_output_test_list)):
                sys_output_file.write("array:" + str(i) + " " + self.sys_output_test_list[i] + "\n")
        sys_output_file.close()

    # =====================================================================
    # print_accuracy:
    # Called by main.
    # Prints confusion matrix and accuracy scores to standard out.
    # =====================================================================

    def print_accuracy(self):
        print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n")
        print("            ", end="")
        for class_name in self.class_list:
            print(" " + class_name, end="")
        print()
        train_correct_count = 0
        for i in range(0, len(self.class_list)):
            print(self.class_list[i], end="")
            for j in range(0, len(self.class_list)):
                print(" " + str(self.train_confusion_matrix[i][j]), end="")
                if i == j:
                    train_correct_count += self.train_confusion_matrix[i][j]
            print()
        train_acc = train_correct_count / len(self.train_vector_list)
        print("\n Training accuracy="+str(train_acc))
        print("\n\nConfusion matrix for the test data:\nrow is the truth, column is the system output\n")
        print("            ", end="")
        for class_name in self.test_class_list:
            print(" " + class_name, end="")
        print()
        test_correct_count = 0
        for i in range(0, len(self.test_class_list)):
            print(self.test_class_list[i], end="")
            for j in range(0, len(self.test_class_list)):
                print(" " + str(self.test_confusion_matrix[i][j]), end="")
                if i == j:
                    test_correct_count += self.test_confusion_matrix[i][j]
            print()
        if len(self.test_vector_list) == 0:
            test_acc = 0.0
        else:
            test_acc = test_correct_count / len(self.test_vector_list)
        print("\n Test accuracy="+str(test_acc))

# =====================================================================
# main:
# =====================================================================


def main():

    start_time = time.perf_counter()

    my_knn = KNN()
    my_knn.read_train_data()
    my_knn.read_test_data()
    my_knn.prep_data_structures()
    my_knn.calc_vector_lengths()
    my_knn.run_test()
    my_knn.print_output_files()
    my_knn.print_accuracy()

    end_time = time.perf_counter()
    # print("runtime: ", end_time - start_time)

    # python build_kNN.py train.vectors.txt test.vectors.txt 1 1 my_sys_output


if __name__ == "__main__":
    main()


# ###################################################################################################
#
# End of file
#
# ###################################################################################################

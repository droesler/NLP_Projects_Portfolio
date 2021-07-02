#!/usr/bin/env python3

# ###################################################################################################
# 
# David Roesler (roeslerdavidthomas@gmail.com) "beamsearch_maxent.py"
#
# ###################################################################################################
#
# Beam search POS tagger
#
# This script implements the beam search for POS tagging.
#
# The format for launching the script is:
# beamsearch_maxent.py test_data boundary_file model_file sys_output beam_size topN topK
#
# ###################################################################################################

import sys          # read from std in
# import re         # used to process input files
import time
import math         # used for log function
import numpy as np  # used for argpartition sort


class Beam:
    test_filename = sys.argv[1]
    boundary_filename = sys.argv[2]
    model_filename = sys.argv[3]
    sys_output_filename = sys.argv[4]
    beam_size = float(sys.argv[5])
    top_n = int(sys.argv[6])
    top_k = int(sys.argv[7])

    test_vector_list = []   # [test instance id] = [[feat1,count],[feat2,count],...]

    class_set = set()       # set of training classes
    class_list = []         # indexed list of training class labels [class id] = class name
    class_to_id = {}        # [class name] = class id (index in class_list)

    test_class_set = set()  # set of test classes not found in training data
    test_class_list = []    # indexed list of training class labels followed by new test class labels
    test_class_to_id = {}   # [test class name] = id number (index in test_class_list)

    feat_set = set()
    feat_to_id = {}         # [feature name] = id number (index in feat_list)
    feat_list = []          # indexed list of features [feature id] = feature name

    class_feat_weight_matrix = []   # [class_id][feature id] = lambda weight (float)
    class_default_weight_list = []  # [class_id] = default lambda weight (float)

    test_confusion_matrix = []  # [true test class id][system test class id] = test instance count

    boundary_list = []              # [test sentence number] = word length of sentence

    num_of_classes = 0
    num_of_feats = 0
    num_of_test_instances = 0

    sys_output_test_list = []   # list of test strings to write to sys_output file. index matches instance id.

    temp_result_list = []       # temporary result list for storing conditional class probabilities p(y|x)
    result_triplet_list = []    # stores top n cond prob results in pairs [class id, cond prob, total log prob]

    inst_class_weight_sum_matrix = []   # [inst id][class id] = sum of weights in input file
    inst_node_list = []                 # [inst_id] = [Beam node1, Beam node 2, ...]

    #  ===========================================================================================
    #  Default constructor.  Initializes data members.
    #  ===========================================================================================

    def __init__(self):
        # self.children = []
        self.parent = None
        self.current_class = "BOS"
        self.prev_current_class = "BOS+BOS"
        self.total_log_prob = 0.0
        self.node_prob = 1.0

    # =====================================================================
    # read_model_file:
    # Called from main. Reads weight values from model file.
    # =====================================================================

    def read_model_file(self):
        with open(Beam.model_filename, mode="r") as input_file:  # open training file
            content_list = input_file.read().splitlines(False)
            content_length = len(content_list)
            line_index = 0
            Beam.class_list.append(content_list[line_index][19:])   # get 1st class data
            Beam.class_set.add(content_list[line_index][19:])
            line_index += 1
            Beam.class_default_weight_list.append(float(content_list[line_index].split()[1]))
            line_index += 1                                                         # ^- read default class weight
            temp_weight_list = []
            while content_list[line_index][0] == " ":  # while first character is ws (feats are indented one ws)
                temp_pair = content_list[line_index].split()
                Beam.feat_list.append(temp_pair[0])
                Beam.feat_set.add(temp_pair[0])
                temp_weight_list.append(float(temp_pair[1]))
                line_index += 1
            Beam.class_feat_weight_matrix.append(temp_weight_list)
            while line_index < content_length and content_list[line_index]:  # get data for any remaining classes
                Beam.class_list.append(content_list[line_index][19:])
                Beam.class_set.add(content_list[line_index][19:])
                line_index += 1
                Beam.class_default_weight_list.append(float(content_list[line_index].split()[1]))
                line_index += 1  # ^- read default class weight
                temp_weight_list = []
                while line_index < content_length and content_list[line_index] and content_list[line_index][0] == " ":
                    temp_weight_list.append(float(content_list[line_index].split()[1]))
                    line_index += 1
                Beam.class_feat_weight_matrix.append(temp_weight_list)
        input_file.close()

    # =====================================================================
    # read_test_data:
    # Called from main. Reads instances and features
    # from test file data.
    # =====================================================================

    def read_test_data(self):
        Beam.test_class_list = Beam.class_list.copy()             # copy training classes to test class list
        with open(Beam.test_filename, mode="r") as input_file:     # open training file
            content_list = input_file.read().splitlines(False)
            line_index = 0
            for line in content_list:
                if line:
                    word_list = line.split()            # split on whitespace
                    if word_list[1] not in Beam.class_set and word_list[1] not in Beam.test_class_set:
                        Beam.test_class_list.append(word_list[1])      # check if class label has been seen
                        Beam.test_class_set.add(word_list[1])
                    # instance name and true class are first items in list
                    Beam.test_vector_list.append([word_list[0], word_list[1]])
                    for i in range(2, len(word_list), 2):
                        if word_list[i] in Beam.feat_set:
                            Beam.test_vector_list[line_index].append(word_list[i])
                            # only retain features that are in training data

                line_index += 1
        input_file.close()

    # =====================================================================
    # read_boundary_data:
    # Called from main. Reads data from boundary file.
    # =====================================================================

    def read_boundary_data(self):
        with open(Beam.boundary_filename, mode="r") as input_file:     # open training file
            content_list = input_file.read().splitlines(False)
            for line in content_list:
                if line:
                    Beam.boundary_list.append(int(line))
        input_file.close()

    # =====================================================================
    # prep_data_structures:
    # Called from main. Initializes and fills data structures that
    # facilitate operations.
    # =====================================================================

    def prep_data_structures(self):
        Beam.num_of_classes = len(Beam.class_list)
        Beam.num_of_feats = len(Beam.feat_list)
        Beam.num_of_test_instances = len(Beam.test_vector_list)

        for i in range(0, Beam.num_of_feats):
            Beam.feat_to_id[Beam.feat_list[i]] = i    # fill feat_to_id dictionary
        for i in range(0, Beam.num_of_classes):
            Beam.class_to_id[Beam.class_list[i]] = i    # fill class_to_id dictionary
        for i in range(0, len(Beam.test_class_list)):
            Beam.test_class_to_id[Beam.test_class_list[i]] = i  # fill test_class_to_id dictionary

        for i in range(0, len(Beam.test_vector_list)):    # replace class names with ids in test instance list
            Beam.test_vector_list[i][1] = Beam.test_class_to_id[Beam.test_vector_list[i][1]]

        Beam.sys_output_test_list = [None] * Beam.num_of_test_instances     # initialize to size of test instance list
        Beam.temp_result_list = [0.0] * Beam.num_of_classes         # list of floats

        for inst_index in range(0, Beam.num_of_test_instances):
            for feat_index in range(2, len(Beam.test_vector_list[inst_index])):
                feat_id_value = Beam.feat_to_id[Beam.test_vector_list[inst_index][feat_index]]
                # replace feat strings with feat ids in original test vector list
                Beam.test_vector_list[inst_index][feat_index] = feat_id_value

        Beam.inst_class_weight_sum_matrix = \
            [[0.0 for column in range(0, Beam.num_of_classes)] for row in range(0, Beam.num_of_test_instances)]

        Beam.test_confusion_matrix = \
            [[0 for column in range(0, len(self.test_class_list))] for row in range(0, len(self.test_class_list))]

        # Build a matrix of weight sums for each word vector
        for inst_id in range(0, Beam.num_of_test_instances):
            for class_id in range(0, Beam.num_of_classes):  # for each class
                # init sum to default weight for class
                Beam.inst_class_weight_sum_matrix[inst_id][class_id] = Beam.class_default_weight_list[class_id]
                for feat_id in Beam.test_vector_list[inst_id][2:]:
                    Beam.inst_class_weight_sum_matrix[inst_id][class_id] += \
                        Beam.class_feat_weight_matrix[class_id][feat_id]

    # =====================================================================
    # build_tree:
    # Called from main. Builds tree for beam search by calling
    # generate_children and pruning nodes.
    # =====================================================================

    def build_tree(self):
        boundary_index = 0
        first_word_index = 0
        while boundary_index < len(Beam.boundary_list):  # for each test sentence
            current_sent_len = Beam.boundary_list[boundary_index]
            last_word_index = first_word_index + (current_sent_len - 1)
            Beam.inst_node_list = [[] for i in range(current_sent_len)]  # [inst_id] = [Beam node1, Beam node 2, ...]

            # build first set of child nodes using the root node. index passed in is index of the children
            self.generate_children(0, first_word_index)  # first arg is local index, 2nd is global vector index
            Beam.inst_node_list[0] = sorted(Beam.inst_node_list[0],
                                            key=lambda x: x.total_log_prob, reverse=True)[:Beam.top_k]  # prune top k
            adjusted_max_log_prob = Beam.inst_node_list[0][0].total_log_prob - Beam.beam_size
            temp_valid_list = []
            for i in range(0, len(Beam.inst_node_list[0])):
                if Beam.inst_node_list[0][i].total_log_prob >= adjusted_max_log_prob:  # prune for beam size
                    temp_valid_list.append(Beam.inst_node_list[0][i])
            Beam.inst_node_list[0] = temp_valid_list

            # build nodes for each word in sentence
            for word_id in range(1, current_sent_len):
                for node in Beam.inst_node_list[word_id - 1]:
                    node.generate_children(word_id, first_word_index + word_id)
                # prune Beam.inst_node_list[i]
                Beam.inst_node_list[word_id] = sorted(Beam.inst_node_list[word_id], key=lambda x: x.total_log_prob,
                                                      reverse=True)[:Beam.top_k]
                adjusted_max_log_prob = Beam.inst_node_list[word_id][0].total_log_prob - Beam.beam_size
                temp_valid_list = []
                for i in range(0, len(Beam.inst_node_list[word_id])):
                    if Beam.inst_node_list[word_id][i].total_log_prob >= adjusted_max_log_prob:
                        temp_valid_list.append(Beam.inst_node_list[word_id][i])
                Beam.inst_node_list[word_id] = temp_valid_list

            # find best result in the final set of nodes and retrace its parents
            Beam.inst_node_list[current_sent_len - 1] = \
                sorted(Beam.inst_node_list[current_sent_len - 1], key=lambda x: x.total_log_prob,
                       reverse=True)
            best_final_node = Beam.inst_node_list[current_sent_len - 1][0]
            leaf_parent = best_final_node.parent
            true_class_id = Beam.test_vector_list[last_word_index][1]
            class_id_prediction = Beam.test_class_to_id[best_final_node.current_class]
            Beam.test_confusion_matrix[true_class_id][class_id_prediction] += 1  # update confusion matrix
            Beam.sys_output_test_list[last_word_index] = \
                Beam.test_vector_list[last_word_index][0] + " " + \
                Beam.test_class_list[true_class_id] + " " + \
                best_final_node.current_class + " " + str(best_final_node.node_prob)
            while last_word_index > first_word_index:
                last_word_index -= 1
                true_class_id = Beam.test_vector_list[last_word_index][1]
                class_id_prediction = Beam.test_class_to_id[leaf_parent.current_class]
                Beam.test_confusion_matrix[true_class_id][class_id_prediction] += 1  # update confusion matrix
                Beam.sys_output_test_list[last_word_index] = \
                    Beam.test_vector_list[last_word_index][0] + " " + \
                    Beam.test_class_list[true_class_id] + " " + \
                    leaf_parent.current_class + " " + str(leaf_parent.node_prob)
                leaf_parent = leaf_parent.parent

            boundary_index += 1
            first_word_index += current_sent_len

    # =====================================================================
    # generate_children:
    # Called from build_tree. Generates sets of child nodes.
    # Calls calc_prob to determine probability values.
    # First argument is the word index at the sentence level and the
    # second argument is the instance vector id number.
    # =====================================================================

    def generate_children(self, word_index: int, inst_id: int):
        prev_string = "prevT=" + self.current_class
        prev_current_string = "prevTwoTags=" + self.prev_current_class
        temp_vector = []

        if prev_string in Beam.feat_set:
            temp_vector += [self.feat_to_id[prev_string]]
        if prev_current_string in Beam.feat_set:
            temp_vector += [self.feat_to_id[prev_current_string]]

        self.calc_prob(temp_vector, inst_id)  # get top n classes from calc_prob function
        # result_triplet_list stores top n cond prob results in pairs [class id, cond prob, total log prob]

        top_index = len(Beam.inst_node_list[word_index])
        for i in range(0, Beam.top_n):  # create top n child nodes and append to current column of beam search tree
            Beam.inst_node_list[word_index].append(Beam())
            child_class_name = Beam.class_list[Beam.result_triplet_list[i][0]]
            Beam.inst_node_list[word_index][top_index].current_class = child_class_name
            Beam.inst_node_list[word_index][top_index].prev_current_class = self.current_class + "+" + child_class_name
            Beam.inst_node_list[word_index][top_index].node_prob = Beam.result_triplet_list[i][1]
            Beam.inst_node_list[word_index][top_index].total_log_prob = Beam.result_triplet_list[i][2]
            Beam.inst_node_list[word_index][top_index].parent = self
            top_index += 1

    # =====================================================================
    # calc_prob:
    # Called from generate_children.
    # Determines probability values for tree nodes.
    # =====================================================================

    def calc_prob(self, vector_to_test: list, inst_id: int):
        z_value = 0.0
        for class_id in range(0, Beam.num_of_classes):  # for each class
            weight_sum = Beam.inst_class_weight_sum_matrix[inst_id][class_id]  # init sum to shared weight for instance
            for feat_id in vector_to_test:
                weight_sum += Beam.class_feat_weight_matrix[class_id][feat_id]
            # all class weights have been summed
            numerator = math.exp(weight_sum)
            Beam.temp_result_list[class_id] = numerator
            z_value += numerator
        # get list of best n results
        top_n_prob_list = np.argpartition(np.array(Beam.temp_result_list), -Beam.top_n)[-Beam.top_n:]
        Beam.result_triplet_list = []
        for class_id in top_n_prob_list:
            cond_prob = Beam.temp_result_list[class_id] / z_value
            Beam.result_triplet_list += [[class_id, cond_prob, (math.log10(cond_prob) + self.total_log_prob)]]

    # =====================================================================
    # print_output_files:
    # Called by main.
    # Prints sys_output file.
    # =====================================================================

    def print_output_files(self):
        with open(Beam.sys_output_filename, mode="w", newline="\n", encoding="utf-8") as sys_output_file:
            sys_output_file.write("%%%%% test data:\n")
            for i in range(0, len(Beam.sys_output_test_list)):
                sys_output_file.write(Beam.sys_output_test_list[i] + "\n")
        sys_output_file.close()

    # =====================================================================
    # print_accuracy:
    # Called by main.
    # Prints confusion matrix and accuracy scores to standard out.
    # =====================================================================

    def print_accuracy(self):
        print("Confusion matrix for the test data:\nrow is the truth, column is the system output\n")
        print("            ", end="")
        for class_name in Beam.test_class_list:
            print(" " + class_name, end="")
        print()
        test_correct_count = 0
        for i in range(0, len(Beam.test_class_list)):
            print(Beam.test_class_list[i], end="")
            for j in range(0, len(Beam.test_class_list)):
                print(" " + str(Beam.test_confusion_matrix[i][j]), end="")
                if i == j:
                    test_correct_count += Beam.test_confusion_matrix[i][j]
            print()
        if len(Beam.test_vector_list) == 0:
            test_acc = 0.0
        else:
            test_acc = test_correct_count / len(Beam.test_vector_list)
        print("\n Test accuracy="+str(test_acc))

# =====================================================================
# main:
# =====================================================================


def main():

    start_time = time.perf_counter()

    my_beam_search = Beam()
    my_beam_search.read_model_file()
    my_beam_search.read_test_data()
    my_beam_search.read_boundary_data()
    my_beam_search.prep_data_structures()
    my_beam_search.build_tree()
    my_beam_search.print_output_files()
    my_beam_search.print_accuracy()

    end_time = time.perf_counter()
    # print("runtime: ", end_time - start_time)

    # python beamsearch_maxent.py my_sec19_21.txt my_boundary.txt m1.txt my_sys_output 2 5 10


if __name__ == "__main__":
    main()


# ###################################################################################################
#
# End of file
#
# ###################################################################################################

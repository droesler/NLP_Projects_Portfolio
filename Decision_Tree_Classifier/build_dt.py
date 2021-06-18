#!/usr/bin/env python3

# ###################################################################################################
# 
# David Roesler (roeslerdavidthomas@gmail.com) "build_dt.py"
#
# ###################################################################################################
#
# Decision Tree Classifier
#
# This script builds a DT tree from the training data, classifies the training and test data,
# and calculates the accuracy.
#
# The format for launching the script is:
# build_dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file
#
# ###################################################################################################

import sys      # read from std in
import re       # used to process input files
import time
import math     # used for log function


class DTree:

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    max_depth = int(sys.argv[3])
    min_gain = float(sys.argv[4])
    model_filename = sys.argv[5]
    sys_output_filename = sys.argv[6]

    train_vector_list = []
    test_vector_list = []
    feat_set = set()
    feat_to_id = {}             # [feature name] = id number (index in feat_list)
    feat_list = []              # indexed list of features
    inst_feat_matrix = []       # sparse matrix [train_vector_list index (instance id)][feature id] = 0 or 1
    class_set = set()           # set of training classes
    test_class_set = set()      # set of test classes not found in training data
    class_list = []             # indexed list of training class labels
    test_class_list = []        # indexed list of training class labels followed by new test class labels
    class_to_id = {}            # [class name] = id number (index in class_list)
    test_class_to_id = {}       # [test class name] = id number (index in test_class_list)

    train_confusion_matrix = []     # [true training class id][system training class id] = training instance count
    test_confusion_matrix = []      # [true test class id][system test class id] = test instance count

    start_inst_list = []            # training instance id list which enters the root node
    test_start_inst_list = []       # test instance id list which enters the root node

    model_file_output_list = []     # list of strings to write to model file
    sys_output_train_list = []      # list of training strings to write to sys_output file. index matches instance id.
    sys_output_test_list = []       # list of test strings to write to sys_output file. index matches instance id.

    #  ===========================================================================================
    #  Default constructor.  Initializes data members.
    #  ===========================================================================================

    def __init__(self):
        self.is_leaf = False                # marks node as a leaf
        self.children = []                  # used to store two DTree child nodes
        self.path_list = []                 # path history of node
        self.split_feat_id = -1             # stores feature id of feature used to split the node
        self.class_id_prediction = None     # predicted class label id for a leaf node
        self.prob_dist = ""                 # string containing probability distribution for leaf nodes

    # =====================================================================
    # read_train_data:
    # Called from main. Reads instances and features
    # from training file data.
    # =====================================================================

    def read_train_data(self):
        with open(DTree.train_filename, mode="r") as input_file:  # open training file
            content_list = input_file.read().splitlines(False)
            line_index = 0
            for line in content_list:
                if line:
                    word_list = line.split()            # split on whitespace
                    DTree.train_vector_list.append([word_list[0]])      # class name is first item in list
                    if word_list[0] not in DTree.class_set:             # check if class label has been seen
                        DTree.class_list.append(word_list[0])
                        DTree.class_set.add(word_list[0])
                    for i in range(1, len(word_list)):
                        word_match = re.match(r'^([^:]+):', word_list[i])
                        if word_match:
                            DTree.train_vector_list[line_index].append(word_match.group(1))
                            if word_match.group(1) not in DTree.feat_set:   # check if feature has been seen
                                DTree.feat_list.append(word_match.group(1))
                            DTree.feat_set.add(word_match.group(1))
                line_index += 1
        input_file.close()

    # =====================================================================
    # read_test_data:
    # Called from main. Reads instances and features
    # from test file data.
    # =====================================================================

    def read_test_data(self):
        DTree.class_list.sort()                                     # sort training class list alphabetically
        DTree.test_class_list = DTree.class_list.copy()             # copy training classes to test class list
        with open(DTree.test_filename, mode="r") as input_file:     # open training file
            content_list = input_file.read().splitlines(False)
            line_index = 0
            for line in content_list:
                if line:
                    word_list = line.split()            # split on whitespace
                    if word_list[0] not in DTree.class_set and word_list[0] not in DTree.test_class_set:
                        DTree.test_class_list.append(word_list[0])      # check if class label has been seen
                        DTree.test_class_set.add(word_list[0])
                    DTree.test_vector_list.append([word_list[0]])       # class name is first item in list
                    for i in range(1, len(word_list)):
                        word_match = re.match(r'^([^:]+):', word_list[i])
                        if word_match:
                            if word_match.group(1) in DTree.feat_set:
                                DTree.test_vector_list[line_index].append(word_match.group(1))
                                # only retain features that are in training data
                line_index += 1
        input_file.close()

    # =====================================================================
    # prep_data_structures:
    # Called from main. Builds data structures that facilitate tree
    # operations.
    # =====================================================================

    def prep_data_structures(self):
        for i in range(0, len(DTree.feat_list)):
            DTree.feat_to_id[DTree.feat_list[i]] = i    # fill feat_to_id dictionary
        for i in range(0, len(DTree.class_list)):          # fill class_to_id dictionary
            DTree.class_to_id[DTree.class_list[i]] = i
        for i in range(0, len(DTree.test_class_list)):          # fill test_class_to_id dictionary
            DTree.test_class_to_id[DTree.test_class_list[i]] = i

        for i in range(0, len(DTree.train_vector_list)):    # replace class names with ids in training instance list
            DTree.train_vector_list[i][0] = DTree.class_to_id[DTree.train_vector_list[i][0]]
            DTree.start_inst_list.append(i)             # create list of train instance ids that will enter tree root

        for i in range(0, len(DTree.test_vector_list)):    # replace class names with ids in test instance list
            DTree.test_vector_list[i][0] = DTree.test_class_to_id[DTree.test_vector_list[i][0]]
            DTree.test_start_inst_list.append(i)    # create list of test instance ids that will enter tree root

        DTree.sys_output_train_list = [None] * len(DTree.train_vector_list)  # initialize to size of instance list
        DTree.sys_output_test_list = [None] * len(DTree.test_vector_list)  # initialize to size of instance list

        # initialize confusion matrices
        DTree.train_confusion_matrix = \
            [[0 for column in range(0, len(DTree.class_list))] for row in range(0, len(DTree.class_list))]
        DTree.test_confusion_matrix = \
            [[0 for column in range(0, len(DTree.test_class_list))] for row in range(0, len(DTree.test_class_list))]

        # initialize instance (row) x feature (column) matrix values to 0
        DTree.inst_feat_matrix = \
            [[0 for column in range(0, len(DTree.feat_list))] for row in range(0, len(DTree.train_vector_list))]
        for inst_index in range(0, len(DTree.train_vector_list)):
            # fill the matrix with 1s for features that appear in the instance vectors
            for feat_index in range(1, len(DTree.train_vector_list[inst_index])):
                DTree.inst_feat_matrix[inst_index][DTree.feat_to_id[DTree.train_vector_list[inst_index][feat_index]]] \
                    = 1

    # =====================================================================
    # build_tree:
    # Called by main.
    # Wrapper function that calls recursive add_children function.
    # =====================================================================

    def build_tree(self):
        if DTree.max_depth == 0:
            return
        root_ent = self.calc_ent(DTree.start_inst_list, len(DTree.start_inst_list))     # find entropy of root node
        self.add_children(DTree.start_inst_list, 0, root_ent)

    # =====================================================================
    # add_children:
    # Called by build_tree.
    # Builds decision tree recursively using training data.
    # =====================================================================

    def add_children(self, current_instance_list: list, depth: int, node_ent: float):
        if depth < DTree.max_depth and len(current_instance_list) > 0 and node_ent > 0:  # If node is a possible split
            result_tuple = self.find_best_feat(current_instance_list)
            info_gain = node_ent - result_tuple[3]
            # if info gain is >= to min_gan and both children will be populated with instances
            if info_gain >= DTree.min_gain and result_tuple[1] and result_tuple[2]:
                self.split_feat_id = result_tuple[0]        # store feat used to split (used again in testing)
                self.children.extend([DTree(), DTree()])    # add 2 nodes
                self.children[0].path_list = self.path_list.copy()  # write path history to children
                self.children[0].path_list.append(DTree.feat_list[result_tuple[0]])
                self.children[1].path_list = self.path_list.copy()
                self.children[1].path_list.append("!" + DTree.feat_list[result_tuple[0]])
                self.children[1].add_children(result_tuple[2], depth + 1, result_tuple[5])  # build on neg/right node
                self.children[0].add_children(result_tuple[1], depth + 1, result_tuple[4])  # build on pos/left node
                return
        self.is_leaf = True     # This node is a leaf node
        self.generate_leaf_prob(current_instance_list)

    # =====================================================================
    # generate_leaf_prob:
    # Called by add_children.
    # Generates probabilities for leaf nodes using training data.
    # =====================================================================

    def generate_leaf_prob(self, current_instance_list: list):
        # build a path string for model file
        path_string = ""
        for i in range(0, len(self.path_list) - 1):
            path_string += self.path_list[i]
            path_string += "&"
        path_string += self.path_list[-1]
        model_output_list = [path_string, str(len(current_instance_list))]

        # generate probabilities for sys_output and model_file
        highest_class_prob = 0
        most_prob_class = 0
        class_count_list = [0] * len(DTree.class_list)  # sum instances for each class
        for inst_id in current_instance_list:
            class_count_list[DTree.train_vector_list[inst_id][0]] += 1
        for class_id in range(0, len(class_count_list)):
            model_output_list.append(DTree.class_list[class_id])  # get class name
            if len(current_instance_list) > 0:
                class_prob = class_count_list[class_id] / len(current_instance_list)
            else:
                class_prob = 0
            if class_prob > highest_class_prob:  # find highest class probability given the current leaf node
                highest_class_prob = class_prob
                most_prob_class = class_id
            model_output_list.append(str(class_prob))

        # store results in model_file_output_list and sys_output_train list
        DTree.model_file_output_list.append(" ".join(model_output_list))
        self.prob_dist = "\t".join(model_output_list[2:])  # store prob dist in leaf node as a string
        for inst_id in current_instance_list:
            DTree.sys_output_train_list[inst_id] = self.prob_dist
            # fill confusion matrix
            DTree.train_confusion_matrix[DTree.train_vector_list[inst_id][0]][most_prob_class] += 1

        # set class prediction for the leaf node
        self.class_id_prediction = most_prob_class

    # =====================================================================
    # find_best_feat:
    # Called by add_children.
    # Finds the feature split that results in the lowest entropy.
    # Returns a tuple containing information to be passed to child nodes.
    # =====================================================================

    def find_best_feat(self, inst_list: list) -> tuple:
        best_avg_ent = 999999999
        best_pos_ent = 999999999
        best_neg_ent = 999999999
        best_feat = -1
        best_pos_list = []
        best_neg_list = []
        inst_count = len(inst_list)

        for feature_id in range(0, len(DTree.feat_list)):   # for each known training feature
            positive_list = []
            negative_list = []
            for instance_id in inst_list:                               # for each instance at this node
                if DTree.inst_feat_matrix[instance_id][feature_id]:     # create two lists based on feature
                    positive_list.append(instance_id)
                else:
                    negative_list.append(instance_id)
            pos_list_length = len(positive_list)
            neg_list_length = len(negative_list)
            pos_entropy = self.calc_ent(positive_list, pos_list_length)
            neg_entropy = self.calc_ent(negative_list, neg_list_length)
            avg_entropy = ((pos_list_length * pos_entropy) + (neg_list_length * neg_entropy)) / inst_count
            if avg_entropy < best_avg_ent:
                best_avg_ent = avg_entropy
                best_feat = feature_id
                best_pos_ent = pos_entropy
                best_neg_ent = neg_entropy
        for instance_id in inst_list:   # re-collect pos and neg lists to avoid repeated copying
            if DTree.inst_feat_matrix[instance_id][best_feat]:
                best_pos_list.append(instance_id)
            else:
                best_neg_list.append(instance_id)
        return best_feat, best_pos_list, best_neg_list, best_avg_ent, best_pos_ent, best_neg_ent

    # =====================================================================
    # calc_ent:
    # Called by build_tree and find_best_feat.
    # Calculates the entropy of a collection of training instances.
    # Returns a float value.
    # =====================================================================

    def calc_ent(self, inst_list: list, total_inst_count: int) -> float:
        class_count_list = [0] * len(DTree.class_list)
        entropy = 0
        for inst_id in inst_list:
            class_count_list[DTree.train_vector_list[inst_id][0]] += 1  # sum instances for each class
        for class_count in class_count_list:
            if class_count:
                class_prob = class_count / total_inst_count
                entropy += (-class_prob * math.log2(class_prob))
        return entropy

    # =====================================================================
    # run_test:
    # Called by main.
    # Wrapper function that calls test_recursive function.
    # =====================================================================

    def run_test(self):
        if len(DTree.test_start_inst_list) == 0:
            return
        self.test_recursive(DTree.test_start_inst_list)     # pass list of test instance ids to recursive test function

    # =====================================================================
    # test_recursive:
    # Called by run_test.
    # Uses pre-built tree to find most likely labels for test data.
    # =====================================================================

    def test_recursive(self, inst_list: list):
        if self.is_leaf:    # stopping condition when reaching a leaf node
            for inst_id in inst_list:
                DTree.sys_output_test_list[inst_id] = self.prob_dist
                # fill test confusion matrix
                DTree.test_confusion_matrix[DTree.test_vector_list[inst_id][0]][self.class_id_prediction] += 1
            return
        pos_list = []
        neg_list = []
        target_word = DTree.feat_list[self.split_feat_id]
        for instance_id in inst_list:
            if target_word in DTree.test_vector_list[instance_id]:
                pos_list.append(instance_id)
            else:
                neg_list.append(instance_id)
        self.children[1].test_recursive(neg_list.copy())    # recursive calls
        self.children[0].test_recursive(pos_list.copy())

    # =====================================================================
    # print_output_files:
    # Called by main.
    # Prints model file.
    # =====================================================================

    def print_output_files(self):
        with open(DTree.model_filename, mode="w", newline="\n", encoding="utf-8") as model_file:
            for line in DTree.model_file_output_list:
                model_file.write(line+"\n")
        model_file.close()
        with open(DTree.sys_output_filename, mode="w", newline="\n", encoding="utf-8") as sys_output_file:
            sys_output_file.write("%%%%% training data:\n")
            for i in range(0, len(DTree.sys_output_train_list)):
                sys_output_file.write("array:" + str(i) + " " + DTree.sys_output_train_list[i] + "\n")
            sys_output_file.write("\n\n%%%%% test data:\n")
            for i in range(0, len(DTree.sys_output_test_list)):
                sys_output_file.write("array:" + str(i) + " " + DTree.sys_output_test_list[i] + "\n")
        sys_output_file.close()

    # =====================================================================
    # print_accuracy:
    # Called by main.
    # Prints confusion matrix and accuracy scores to standard out.
    # =====================================================================

    def print_accuracy(self):
        print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n")
        print("            ", end="")
        for class_name in DTree.class_list:
            print(" " + class_name, end="")
        print()
        train_correct_count = 0
        for i in range(0, len(DTree.class_list)):
            print(DTree.class_list[i], end="")
            for j in range(0, len(DTree.class_list)):
                print(" " + str(DTree.train_confusion_matrix[i][j]), end="")
                if i == j:
                    train_correct_count += DTree.train_confusion_matrix[i][j]
            print()
        train_acc = train_correct_count / len(DTree.train_vector_list)
        print("\n Training accuracy="+str(train_acc))
        print("\n\nConfusion matrix for the test data:\nrow is the truth, column is the system output\n")
        print("            ", end="")
        for class_name in DTree.test_class_list:
            print(" " + class_name, end="")
        print()
        test_correct_count = 0
        for i in range(0, len(DTree.test_class_list)):
            print(DTree.test_class_list[i], end="")
            for j in range(0, len(DTree.test_class_list)):
                print(" " + str(DTree.test_confusion_matrix[i][j]), end="")
                if i == j:
                    test_correct_count += DTree.test_confusion_matrix[i][j]
            print()
        if len(DTree.test_vector_list) == 0:
            test_acc = 0.0
        else:
            test_acc = test_correct_count / len(DTree.test_vector_list)
        print("\n Test accuracy="+str(test_acc))

# =====================================================================
# main:
# =====================================================================


def main():

    root = DTree()
    root.read_train_data()
    root.read_test_data()
    root.prep_data_structures()
    root.build_tree()
    root.run_test()
    root.print_output_files()
    root.print_accuracy()


if __name__ == "__main__":
    main()


# ###################################################################################################
#
# End of file
#
# ###################################################################################################

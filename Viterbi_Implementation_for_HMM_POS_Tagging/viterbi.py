#!/usr/bin/env python3

# ###################################################################################################
# 
# David Roesler (roeslerdavidthomas@gmail.com) "viterbi.py"
#
# ###################################################################################################
#
# Viterbi Implementation
#
# This script reads an HMM file and an input text, then finds the most probable tag sequence for
# the text and outputs the results to another file.
#
# The format for launching the script is: viterbi.py input_hmm test_file output_file
#
# ###################################################################################################

import sys  # command line arguments
from collections import defaultdict  # used to initialize dict values to 0
import math  # used for log function

# =====================================================================
# Viterbi (class):
# =====================================================================


class Viterbi:

    #  ===========================================================================================
    #  Default constructor.  Initializes data members.
    #  ===========================================================================================

    def __init__(self):
        self.content_list = []                                  # HMM input file content
        self.init_dict = defaultdict(lambda: 1.0)               # [state] = log prob (1 represents zero probability)
        self.trans_dict = defaultdict(lambda: defaultdict(lambda: 1.0))     # [from state][to state] = log prob
        self.trans_dict2 = defaultdict(lambda: defaultdict(lambda: 1.0))    # [to_state_id][from_state_id] = log prob
        self.emit_dict = defaultdict(lambda: defaultdict(lambda: 1.0))      # [symbol][state] = log prob
        self.emit_dict2 = defaultdict(lambda: defaultdict(lambda: 1.0))    # [symbol_id][state_id] = log prob
        self.trans_array = []       # full matrix [from_state][to_state] = log prob (1.0 is value for 0 prob)
        self.emit_array = []        # full matrix [symbol][state] = log prob (1.0 is value for 0 prob)
        self.symbol_set = set()
        self.symbol_list = []
        self.state_set = set()
        self.state_list = []
        self.state_range = []   # list of integers from 0 to the state count (used to iterate across states ids)
        self.state_count = 0
        self.state_to_id = {}   # [state] = state_id
        self.symbol_to_id = {}  # [symbol] = symbol_id

    # =====================================================================
    # read_input_data:
    # Called from main. Extracts data from HMM file.
    # =====================================================================

    def read_input_data(self, input_filename: str):
        with open(input_filename, 'r') as input_file:  # read test file
            self.content_list = [i.strip() for i in input_file.read().splitlines(False)]
            content_length = len(self.content_list)
            index = 5
            while index < content_length and self.content_list[index] != "\\init":
                index += 1
            index += 1
            while index < content_length and self.content_list[index]:
                if not self.add_init_line(self.content_list[index].split()):
                    sys.stderr.write("warning: the prob is not in [0,1] range: "+self.content_list[index])
                index += 1
            while index < content_length and self.content_list[index] != "\\transition":
                index += 1
            index += 1
            while index < content_length and self.content_list[index]:
                if not self.add_trans_line(self.content_list[index].split()):
                    sys.stderr.write("warning: the prob is not in [0,1] range: "+self.content_list[index])
                index += 1
            while index < content_length and self.content_list[index] != "\\emission":
                index += 1
            index += 1
            while index < content_length and self.content_list[index]:
                if not self.add_emit_line(self.content_list[index].split()):
                    sys.stderr.write("warning: the prob is not in [0,1] range: "+self.content_list[index])
                index += 1
        input_file.close()

    # =====================================================================
    # add_init_line:
    # Called from read_input_data. Adds input initialization line to the
    # HMM. Returns false if prob value is out of 0 -> 1 range.
    # =====================================================================

    def add_init_line(self, to_add_list: list) -> bool:
        init_prob = float(to_add_list[1])
        if init_prob < 0.0 or init_prob > 1.0:
            return False
        elif init_prob > 0.0:  # for prob == 0, default dictionary log prob is 1.0
            self.init_dict[to_add_list[0]] = math.log10(init_prob)
        return True

    # =====================================================================
    # add_trans_line:
    # Called from read_input_data. Adds input transition line to the
    # HMM. Returns false if prob value is out of 0 -> 1 range.
    # =====================================================================

    def add_trans_line(self, to_add_list: list) -> bool:
        trans_prob = float(to_add_list[2])
        if trans_prob < 0.0 or trans_prob > 1.0:
            return False
        else:
            if trans_prob > 0.0:  # for prob == 0, default dictionary log prob is 1.0
                self.trans_dict[to_add_list[0]][to_add_list[1]] = math.log10(trans_prob)
            self.state_set.add(to_add_list[0])
            self.state_set.add(to_add_list[1])
        return True

    # =====================================================================
    # add_emit_line:
    # Called from read_input_data. Adds input emission line to the
    # HMM. Returns false if prob value is out of 0 -> 1 range.
    # =====================================================================

    def add_emit_line(self, to_add_list: list) -> bool:
        emit_prob = float(to_add_list[2])
        if emit_prob < 0.0 or emit_prob > 1.0:
            return False
        else:
            if emit_prob > 0.0:
                self.emit_dict[to_add_list[1]][to_add_list[0]] = math.log10(emit_prob)
            self.symbol_set.add(to_add_list[1])
        return True

    # =====================================================================
    # build_hmm:
    # Called from main. Builds structure of HMM based on input file data.
    # =====================================================================

    def build_hmm(self):
        self.state_list = list(self.state_set)      # convert state set to indexed list
        self.symbol_list = list(self.symbol_set)    # convert symbol set to indexed list
        self.state_count = len(self.state_list)     # store the number of HMM states
        self.state_range = range(0, self.state_count)

        for index in range(0, len(self.state_list)):            # fill state_to_id dictionary
            self.state_to_id[self.state_list[index]] = index
        for index in range(0, len(self.symbol_list)):           # fill symbol_to_id dictionary
            self.symbol_to_id[self.symbol_list[index]] = index

        for from_state in self.trans_dict:   # build reverse order trans_dict2 [to_state_id][from_state_id] = log prob
            for to_state in self.trans_dict[from_state]:
                self.trans_dict2[self.state_to_id[to_state]][self.state_to_id[from_state]] = \
                    self.trans_dict[from_state][to_state]

        for symbol in self.emit_dict:   # build emit_dict2 [symbol_id][state_id] = log prob
            for state in self.emit_dict[symbol]:
                self.emit_dict2[self.symbol_to_id[symbol]][self.state_to_id[state]] = \
                    self.emit_dict[symbol][state]

        # create full 2d transition and emission arrays for all known states and symbols
        self.trans_array = [[1.0 for to_state in self.state_list] for from_state in self.state_list]
        for to_state in self.trans_dict2:
            for from_state in self.trans_dict2[to_state]:
                self.trans_array[from_state][to_state] = self.trans_dict2[to_state][from_state]

        self.emit_array = [[1.0 for state in self.state_list] for symbol in self.symbol_list]
        for symbol in self.emit_dict2:
            for state in self.emit_dict2[symbol]:
                self.emit_array[symbol][state] = self.emit_dict2[symbol][state]

    # =====================================================================
    # process_test_data:
    # Called from main. Writes state sequence for the given test data to
    # the output file.
    # =====================================================================

    def process_test_data(self, test_filename: str, output_filename: str):
        with open(test_filename, 'r') as test_file:  # read test file
            with open(output_filename, mode="w", newline="\n", encoding="utf-8") as output_file:  # open output file
                test_lines = test_file.read().splitlines(False)
                for line in test_lines:                                             # for each line in test file
                    output_file.write(line+" =>")
                    test_word_list = line.split()
                    for i in range(0, len(test_word_list)):
                        if test_word_list[i] not in self.symbol_set:                # replace unknown symbols with <unk>
                            test_word_list[i] = "<unk>"
                        test_word_list[i] = self.symbol_to_id[test_word_list[i]]    # convert symbols to id numbers
                    best_sequence = self.find_best_sequence(test_word_list)
                    for j in range(0, len(best_sequence)):
                        output_file.write(" " + best_sequence.pop())
                    output_file.write("\n")
            output_file.close()
        test_file.close()

    # =====================================================================
    # find_best_sequence:
    # Called from process_test_data. Finds the most probable state sequence.
    # =====================================================================

    def find_best_sequence(self, word_list: list) -> list:  # word_list is a list of symbol id numbers
        sentence_length = len(word_list)

        # build delta log prob matrix with default values of 1.0 (zero probability) and -1 (dummy back pointer)
        delta = [[[1.0, -1] for column in range(0, sentence_length+1)] for row in self.state_range]
        for init_state in self.init_dict:   # fill first column of delta with initial log prob values
            delta[self.state_to_id[init_state]][0][0] = self.init_dict[init_state]

        for t, word in enumerate(word_list):  # 1st loop: 'word' is actually t+1 since word_list is offset by BOS
            for to_state in self.emit_dict2[word]:  # 2nd nested loop
                emit_value = self.emit_array[word][to_state]
                result_list = []
                for from_state in self.trans_dict2[to_state]:           # 3rd nested loop
                    if delta[from_state][t][0] < 1.0:                   # only calculate non-dead paths
                        trans_value = self.trans_array[from_state][to_state]
                        result_list.append(((trans_value + emit_value + delta[from_state][t][0]), from_state))
                if result_list:
                    best_result = max(result_list, key=lambda i: i[0])
                    delta[to_state][t + 1][0] = best_result[0]          # set path log prob in delta element
                    delta[to_state][t + 1][1] = best_result[1]          # set back pointer in delta element
        best_end_state_list = []
        for m in self.state_range:                                   # find best path ending
            current_log_prob = delta[m][sentence_length][0]     # get log prob of each state in last column
            if current_log_prob < 1.0:
                best_end_state_list.append((current_log_prob, m))
        best_end_index = max(best_end_state_list, key=lambda i: i[0])[1]    # best_end_index is the best back pointer
        best_sequence = [str(delta[best_end_index][sentence_length][0])]    # init result list with total log prob
        for n in reversed(range(1, sentence_length+1)):                     # follow back pointers through the sequence
            best_sequence.append(self.state_list[best_end_index])
            best_end_index = delta[best_end_index][n][1]
        best_sequence.append(self.state_list[best_end_index])
        return best_sequence

# =====================================================================
# main:
# =====================================================================


def main():
    input_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_filename = sys.argv[3]
    my_viterbi = Viterbi()    # create instance of class

    my_viterbi.read_input_data(input_filename)
    my_viterbi.build_hmm()
    my_viterbi.process_test_data(test_filename, output_filename)



if __name__ == "__main__":
    main()


# ###################################################################################################
#
# End of file
#
# ###################################################################################################

#!/usr/bin/env python3

# ###################################################################################################
#
# David Roesler (roeslerdavidthomas@gmail.com) "language_classifier.py"
#
# ###################################################################################################
#
# Probabilistic Language Identifier
#
# This script is an implementation of a naive Bayesian classifier that classifies fragments of
# text according to language category. The script builds a set of language models from a set of
# training data for each of 15 language types. Log probabilities of each sentence are generated
# and output to the console. The most probable language is output to the console, labeled as "result".
# Results that do not meet a series of confidence thresholds are reported as "unk".
#
# ###################################################################################################


import os  # access to file folders
import glob  # checks file names for desired naming pattern
import re  # reg ex
from typing import TextIO  # for output file type hints
import sys  # for command line arguments
import math


# =====================================================================
# build_dict:
# Builds a language model dictionary from external corpora files.
# Accepts a file object as an argument.
# Returns a tuple containing the dictionary for a single language and
# a type count of the corpus for that language.
# =====================================================================


def build_dict(model_file: TextIO):
    token_count = 0
    to_build = {}
    content = model_file.read()                 # stores file contents in a string
    split_content = content.splitlines(False)   # splits string into lines without linebreaks
    for line in split_content:
        split_line = line.split("\t")
        to_build[split_line[0]] = int(split_line[1])    # stores the type and its count in dictionary
        token_count += int(split_line[1])                # totals the token count for the language
    for entry in to_build:
        to_build[entry] = math.log10((to_build[entry] + .000001) / (token_count + .0015))
        # replaces the type count with log probability for each type
        # using plus k smoothing with a value of .000001
    return to_build, token_count


# =====================================================================
# get_prob:
# Finds the log probability of a sentence string for a given language.
# Accepts the language dictionary, the string to examine, and a token
# count for the language as arguments. Returns a tuple (a pair of floats)
# containing the smoothed log probability for the sentence and the
# percentage of words in the sentence that were found in the language
# dictionary.
# =====================================================================


def get_prob(lang_dict: dict, line: str, token_count: int):
    result = 0.0
    line_split = line[2:].split()   # trims first 2 chars of line and split into words
    word_count = 0
    found_count = 0
    for word in line_split:
        word = re.sub(r"^[\., !¡¥\$£¿;:\(\)\"\'\—\–\-/\[\]¹²³«»]+|[\., !¡¥\$£¿;:\(\)\"\'\—\–\-/\[\]¹²³«»]+$", "", word)
        # trims punctuation from both ends of words
        word_count += 1
        if word in lang_dict:
            result += lang_dict[word]           # gets log probability from dictionary if word is found there
            found_count += 1
        else:
            result += math.log10(.000001/(token_count + .0015))    # assigns log prob. for unknown words
    percent_found = found_count / word_count    # gets percentage of the recognized words in the sentence
    return result, percent_found


# =====================================================================
# main:
# =====================================================================


def main():
    model_dict = {}          # dictionary of all lang model dictionaries and token counts for each language
    log_prob_dict = {}       # dictionary of log probabilities for an input sentence
    percent_found_dict = {}  # dictionary containing percentage of sentence tokens found for each language
    test_path = 'test.txt'

    if len(sys.argv) > 1:    # accepts "extra" as a command line argument to open the "extra-test.txt" file
        if sys.argv[1] == "extra":
            test_path = 'extra-test.txt'

    for filename in glob.glob('/lang_models/*.unigram-lm'):
        with open(os.path.join(os.getcwd(), filename), 'r', encoding='latin-1') as f:
            lang_key = os.path.basename(f.name)[0:3]    # gets language name from file name
            model_dict[lang_key] = build_dict(f)        # builds a multi-language dictionary of tuples (dict, int)
        f.close()
    with open(test_path, 'r', encoding='latin-1') as train_file:
        content = train_file.read()
        split_content = content.splitlines(False)
        for line in split_content:
            print(line)                 # print input sentence
            for key in model_dict:      # for each known language, call get_prob function and store results
                (log_prob_dict[key], percent_found_dict[key]) = get_prob(model_dict[key][0], line, model_dict[key][1])
            for i in log_prob_dict:
                print("%s\t%f" % (i, log_prob_dict[i]))  # print log probability for each language

            sort_prob_tuple_list = sorted(log_prob_dict.items(), key=lambda x: x[1])  # sort log probs by value
            highest_key = sort_prob_tuple_list[-1][0]
            highest_percent = percent_found_dict[highest_key]
            if highest_percent < (1/3):
                print("result unk")     # results where confidence threshold is not reached are reported as "unk"
            else:
                print("result", highest_key)
    train_file.close()


if __name__ == "__main__":
    main()

:warning: (Documentation in progress!):warning:

Naive Bayes Language Classifier
---

This project is a Python implementation of a naive Bayesian classifier that classifies fragments of text according to language category. The script builds a set of 15 unigram language models from a set of files containing token counts for 1,500 of the types in each language. The log probability of each sentence (given a language) is generated and output to the console (see image below). The most probable language is also output to the console and labeled as the “result".
To calculate the (smoothed) log probability of each word in the 15 language samples, the following formula was used:



### About the code

The format for launching the script is:  

```build_dt.sh training_data test_data max_depth min_gain model_file sys_output```

where ```training_data``` is train.vectors.txt, ```test_data``` is test.vectors.txt, ```max_depth``` is the maximum depth of the tree, ```min_gain``` is the minimal information gain for each split, ```model_file``` is the filename for the output model, and ```sys_output``` is the classification results for the train and test data.

| <img src="results_table.png" alt="results_table.png" width="500"/> | 
|:--:| 
| *Decision tree results when min_gain=0.* |



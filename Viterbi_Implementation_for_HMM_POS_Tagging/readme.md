:warning: (Documentation in progress!):warning:

Viterbi Implementation for HMM POS Tagging
---

This script reads an HMM file produced by the [MALLET](http://mallet.cs.umass.edu/) machine learning toolkit and an input text, uses an implementation of the Viterbi algorithm to find the most probable tag sequence for the text, and then outputs the results to a file.

### About the code

The format for launching the script is:  

```viterbi.py input_hmm test_file output_file```

where ```input_hmm``` is a valid Mallet HMM file (such as hmm5, which can be found in hmm5.rar), ```test_file``` is test.word, and ```output_file``` is the desired name of the output file.

| <img src="knn_results.png" alt="knn_results.png" width="500"/> | 
|:--:| 
| *kNN classifier test accuracy by k values and similarity functions.* |

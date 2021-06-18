:warning: (Documentation in progress!):warning:

Naive Bayes Language Classifier
---

This project is a Python implementation of a naive Bayesian classifier that classifies fragments of text according to language category. 

### About the code

The script builds a set of 15 unigram language models from a set of files containing token counts for 1,500 of the types in each language. The log probability of each sentence (given a language) is generated and output to the console (see image below). The most probable language is also output to the console and labeled as the “result".
To calculate the (smoothed) log probability of each word in the 15 language samples, the following formula was used:

<p align="center">
  <img src="formula1.jpg" alt="formula1.jpg" width="400"/>
</p>

Where N is the total token count for the language model and V is the type count for the language model (1,500 for all models). The “add-k” smoothing technique was utilized to account for words that are unknown to the language models. The value of .000001 was assigned to k for these calculations (this choice is discussed in Section 2).
The language category of a given text is labeled as “unk” (unknown) if the percentage of recognized sentence tokens in the most probable language is less than 1/3. This condition was set in order to identify cases where data is sparse and it becomes problematic to maintain a high degree of confidence in any classification.

<br/>
| <img src="output_sample.jpg" alt="output_sample.jpg" width="1000"/> | 
|:--:| 
| *Results for the first input sentence.* |



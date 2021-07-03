:warning: (Documentation in progress!) :warning:

Beamsearch for Maxent POS Tagging
---

This script reads a MaxEnt model file produced by the [MALLET](http://mallet.cs.umass.edu/) ML toolkit and uses an implementation of the beamsearch algorithm to find the most probable tag sequence for the text.

### About the code

The format for launching the script is:  

```beamsearch_maxent.py test_data boundary_file model_file sys_output beam_size topN topK```

where ```input_hmm``` is hmm5, which can be found in hmm5.rar, ```test_file``` is test.word, and ```output_file``` is the desired name of the output file.

| <img src="output_sample.png" alt="output_sample.png" width="1700"/> | 
|:--:| 
| *A sample of the output with the format of: (input) => (trigram POS label) (joint log probability of sequence).* |

:warning: (Documentation in progress!):warning:

Deep Averaging Network
---

A Pytorch implementation of the Deep Averaging Network introduced in Iyyer et al (2015). Performs binary sentiment classification on the [IMDB reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/). 
 
### About the code

Command line parameters:
```
    # model arguments
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=300)
    # training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=572)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--L2', action="store_true")
    # data arguments
    parser.add_argument('--data_dir', type=str, default='/data')
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--padding_index', type=int, default=1)
```


| <img src="DAN_output.png" alt="DAN_output.png" width="1000"/> | 
|:--:| 
| *Results for --num epochs 12 --patience 3 --L2.* |






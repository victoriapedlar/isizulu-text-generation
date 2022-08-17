# isizulu-text-generation

# AWD-LSTM

+ This code was originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model) and is heavily inspired by the original AWD-LSTM implementation [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm)

+ The code in this notebook is available on [google colab](https://colab.research.google.com/drive/1yyUGJfyYKdvPi6J7ZlsxPg9E_ppZG1xU) and on [github](https://github.com/mikkelbrusen/awd-inspired-lstm).

The project was carried out by [Gustav Madslund](https://github.com/gustavmadslund) and [Mikkel Møller Brusen](https://github.com/mikkelbrusen).


### Core components

1.   **[x]  - Multi Layer** - We will need to controll what happens in between the layers, therefore, instead of using the multi layer cuDNN lstm implementation, we will create multiple single layer cuDNN lstms.
2.   **[x] - Weight drop** using DropConnect on hidden-hidden weights $[U^i, U^f, U^o, U^c]$ before forward and backward pass - makes it possible to use cuDNN LSTM
3.   **[x] - Optimization** using SGD and ASGD while training

### Extended regularization techniques
4.   **[ ] - Variable sequence length** to allow all elements in the dataset to experience a full BPTT window
  - **[ ] - Rescale learning rate** to counter the varible sequence lengths favoring short sequences with fixed learning rate
5.   **[x] - Variational dropout AKA LockDrop** for everything else than hidden-hidden, such that we use same dropout mask for all input/output in a forward backward pass of LSTM
6.   **[x] - Embedding dropout** which is **not** just a dropout applied on the embedding
7.   **[x]  - Weight tying** to reduce parameters and prevent model from having to learn one-to-one correspondance between input and output
8.   **[x] - Embed size** independent from hidden size, to reduce parameters.
9.   **[ ] - AR and TAR** - $L_2$-regularization by applying AR and TAR loss on the final RNN layer - can screw stuff up

# wavernn
WaveRNN is a technique used to synthesise neural audio much faster than other neural synthesizer by providing rudimentary improvements.

This implementation is based on tensorflow and will require librosa to manipulate raw audio.


These points provides an intution about the model, how the model has been implented [any inputs on this are most welcome]

 1. Input data is taken as float32 based and normalized to 16bit unsigned data.
 2. 16bit data has been split into two components by using divmod (with floor )
 3. Two different `RNNs` has been created of `cell-length` 896 as mentioned in the paper, being stacked in 2 layers (have to test number of layers).
 4. First rnn that synthesize `coarse_data` doesnt need `c(t)`, so input to this is `[batch_size,sequence_length,2]` where currently batch size is 1 with sequence length of 200 and 2 for `[c(t-1),f(t-1)]`
 5. Second rnn that synthesize `fine_data` needs `c(t)` so currently it is dependent on the `coarse_data` for generation (but it should improve after subscaling). It has an input vector of `[batch_size,sequence_length,3]` where 3 is to store extra `c(t)`
 6. Output of these `RNNs` are parsed to a dense linear transformation of same length, which is then used passed in to `relu`  to remove negative entries.
 7. For cost function, currently `softmax_cross_entropy_with_logits_v2` is used but most probably will switch to sparse one.
 8. Adam is used to optimize the whole network, this initial commit doesnt include intensive hyper parameter testing, so it is using default learning rate.

## Tasks
- [x] Basic implementation, which improves net algorithmic efficiency
- [ ] Providing support for faster future prediction (On-going)
- [x] Transfer from notebook based development to OOPS, for better management (WIP)
- [ ] Sparse Prunification and Sub batched sampling
- [ ] Vocoder Implementation


Please go through the issues, there are many conceptual doubts and I would love to hear opinions on those.

This repository only provides implementation of the model `WaveRNN` as mentioned [here](https://arxiv.org/abs/1802.08435).

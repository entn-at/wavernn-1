class params:

    batch_size = 128
    coarse_features = 2
    fine_features = 3
    max_sequence_length = 100
    cells_detail = [256, 256, 256]
    regression = True
    hidden_units = 256
    output_units = 256
    training_test_ratio = 0.7
    epochs = 1
    segmentation = True
    segmentation_length = 5
    sample_rate = 22050


hyper_parameters = {
    # Total length of sine wave // trained on sine wave to better analyse whole algorithm
    "length_of_wave": 100000,
    "sample_rate": 16000,  # number of samples in a second
    "batch_size": 1,  # number of batches
    # maximum number of sequential steps, this network can generate (should be 1k but it is computationally costly and requires much more training time)
    "max_sequence_length": 100,
    "number_input_in_each_instant": {
        "coarse": 2,  # to incorporate [c(t-1),f(t-1)]
        "fine": 3  # to incorporate [c(t-1),f(t-1),c(t)]
    },
    "number_of_the_rnn_cells": 896,  # Defined in the paper
    "depth_of_relu_networks": 896,  # Defined in the paper
    # By using 256, we can level the output in terms of probability, thus better ways to analyse
    "softmax_probability": 256,
    "number_of_layers_in_rnns": 2,  # Needs to find best layer size
    "new_sequences": 10,  # while generating how many new sequence it should generate
}
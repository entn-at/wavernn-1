

class params:
    epochs = 10
    sample_rate = 22050
    nFft = 1024
    hop_size = 256
    constant_values = 0.
    num_mels = 64
    checkpoint_every = 100
    curr_index = 0
    raw_data_dirs = ["data/speaker/", ]
    train_data_dir = "training_data/"
    log_file_name = "log/wavernn.log/"
    model_file_name = "log/wavernn.model/"
    sequence_units = 256
    prenet_outp = 256
    num_of_workers = 8
    bit_rate_in_power = 16

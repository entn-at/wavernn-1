from librosa.core import load as librosa_import
from librosa.feature import melspectrogram
import numpy as np
from glob import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from wavernn.hparams import params

scale_factor = int(2**(params.bit_rate_in_power-1))
split_factor = int(2**(params.bit_rate_in_power/2))
second_split = int(2**(params.bit_rate_in_power/2-1))
one_hot = np.eye(split_factor)


def postprocess_data(linear, mel, file_name):
    """

        Scale the input into coarse and fine bands, so that
        training doesnot bottle neck on cpu computing these bands
        again and again for every epochs

        Inputs :
            linear:- audio in time dimension [None]
            mel:- audio in mel dimention [frames, None]
            file_name:- name of the file in which current training data is


        Returns :
            None

    """

    num_of_frames = mel.shape[0]
    bit_16_signal = scale_factor*(linear+1)
    coarse_linear, fine_linear = np.divmod(bit_16_signal, split_factor)

    scaled_coarse = coarse_linear/second_split - 1
    padded_coarse = np.insert(scaled_coarse[:-1], 0, 0)
    padded_fine = np.insert((fine_linear/second_split - 1)[:-1], 0, 0)

    np.savez(file_name, scaled_coarse, padded_coarse, padded_fine,
             coarse_linear, fine_linear, mel)

    return None


def process_individual_file(input_file_uri, index):
    """
        It processes each input file and saves those in the required
        format, which is going to be used while training.

        Inputs:
            input_file_name : name of the data file to be used during training.

        Returns:
            {output_file_uri, num_of_mel_frames, length_of_time_scaled_audio}

    """

    data = [librosa_import(uri)[0] for uri in files_uri][0]

    mfcc = melspectrogram(data,
                          n_fft=params.nFft,
                          hop_length=params.hop_size,
                          n_mels=params.num_mels).T

    mfcc_shape = mfcc.shape
    num_elem_in_mfcc = mfcc_shape[0]*mfcc_shape[1]
    pad_val = params.scale_factor*num_elem_in_mfcc - data.shape[0]
    data = np.pad(data, [0, pad_val], mode="constant")
    assert data.shape[0] % num_elem_in_mfcc == 0

    output_file_uri = os.path.join(
        training_data_folder, "sample_{}".format(index))
    postprocess_data(data, mfcc, output_file_uri)

    return {output_file_uri, mfcc_shape[0], len(data)}


def preprocess(input_folders, out_dir):
    concurrent_executor = ThreadPoolExecutor(max_workers=params.num_of_workers)

    source_files = [glob(source_folder+"/*.wav")
                    for source_folder in input_folders]

    number_of_files = list(map(len, source_files))
    min_constraint = np.min(number_of_files)

    shuffled_entries = []
    for index, each_file_len in enumerate(number_of_files):
        np.random.shuffle(source_files[index])
        shuffled_entries.append(source_files[index][:min_constraint])
    shuffled_array = np.asarray(shuffled_entries).T
    indices = np.arange(len(shuffled_array))

    meta_data = open(training_data_folder +
                     "/{}".format("metadata.txt"), mode, buffering)
    for e in tqdm(concurrent_executor.map(process_individual_file, shuffled_array, indices)):
        meta_data.write(e)
    meta_data.flush()


def run_preprocess():

    global training_data_folder
    training_data_folder = params.train_data_dir
    os.makedirs(training_data_folder, exist_ok=True)

    input_folders = params.input_data_dir

    preprocess(input_folders)


def main():
    run_preprocess()


if __name__ == '__main__':
    main()

import tensorflow as tf
import scipy.io.wavfile as siowav
import librosa
import numpy as np
import pickle as pkl
import os
import argparse
import tqdm
import random
import codecs


def get_arguments():
    parser = argparse.ArgumentParser(description="Convert wav file to TFRecords file.")

    parser.add_argument("--wav_dir", "-w", type=str, default="./assets/wav", help="")
    parser.add_argument("--cond_dir", "-a", type=str, default="./assets/aco_feat/mel", help="")
    parser.add_argument("--key_path", "-k", type=str, default="./assets/keys/train.txt", help="")
    parser.add_argument("--target_path", "-d", type=str, default="./assets/tfrecords/train.tfrecords", help="")
    parser.add_argument("--sr", type=int, default=16000, help="")
    parser.add_argument("--cond_dims", type=int, default=80, help="")

    return parser.parse_args()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_to_bytes(key, wav_dir, cond_dir, sr, cond_dims):
    wav_path = os.path.join(wav_dir, key + ".wav")
    cond_path = os.path.join(cond_dir, key + ".bin")

    this_sr, _ = siowav.read(wav_path)  # just for get this_sr
    this_wav, _ = librosa.core.load(wav_path, this_sr)
    if this_sr != sr:
        this_wav = librosa.core.resample(this_wav, this_sr, sr) # [-1, 1]

    cond = np.reshape(np.fromfile(cond_path, dtype=np.float32), (-1, cond_dims))
    cond_frames = len(cond)

    key_raw = key.encode("utf-8")
    this_wav *= 32768.  # for i16
    wav_raw = this_wav.astype(np.int16).tostring()  # little-endian
    cond_raw = cond.astype(np.float32).tostring()
    # create tf example feature
    example = tf.train.Example(features=tf.train.Features(feature={
        "sr": _int64_feature(int(sr)),
        "key_raw": _bytes_feature(key_raw),
        "wav_raw": _bytes_feature(wav_raw),
        "frames": _int64_feature(int(cond_frames)),
        "cond": _bytes_feature(cond_raw)}))
    return example.SerializeToString()

def get_key_lst(key_path):
    with codecs.open(key_path, "r", "utf-8") as f:
        key_lst = [line.strip() for line in f.readlines()]
    return key_lst

def main():
    print(__file__)
    args = get_arguments()

    # 1st. get path_lst, which contains all the paths to wav files.
    #key_lst = get_key_lst(args.cond_dir)
    key_lst = get_key_lst(args.key_path)
    assert key_lst, "[E] Key list is empty!"

    # 5th. extract features, normalize them and write them back to disk.
    count_not_found = 0
    with tf.python_io.TFRecordWriter(args.target_path) as writer:
        for key in tqdm.tqdm(key_lst):
            try:
                example_str = read_to_bytes(key=key, wav_dir=args.wav_dir, cond_dir=args.cond_dir,
                                            sr=args.sr, cond_dims=args.cond_dims)
                writer.write(example_str)
            except ValueError as e:
                print(e, key)
            except FileNotFoundError as e:
                print(e, key)
                count_not_found += 1
    print("{} files not found.".format(count_not_found))

    print("Congratulations!")


if __name__ == "__main__":
    main()

import tensorflow as tf
import scipy.io.wavfile as siowav
import librosa
import numpy as np
import pickle as pkl
import os
import argparse
import tqdm
import random
from utils.audio import get_mel, get_mel_filterbank, get_stftm, log_compress, preemphasis


def get_arguments():
    parser = argparse.ArgumentParser(description="Get mel-spectrogram from WAV file!")

    parser.add_argument("--src_dir", "-s", type=str, default="./assets/wav", help="")
    parser.add_argument("--dest_dir", "-d", type=str, default="./assets/aco_feat/mel", help="")
    parser.add_argument("--sr", type=int, default=16000, help="")
    parser.add_argument("--frame_shift", type=float, default=0.0125, help="")
    parser.add_argument("--frame_size", type=float, default=0.050, help="")
    parser.add_argument("--n_fft", type=int, default=1024, help="")
    parser.add_argument("--n_mels", type=int, default=80, help="")
    parser.add_argument("--window", type=str, default="hann", help="")
    parser.add_argument("--floor_gate", type=float, default=1e-5, help="")

    args = parser.parse_args()
    return args


def wav2mel(key, root, sr, frame_shift, frame_size, n_fft, window, mel_filterbank, floor_gate):
    wav_path = os.path.join(root, key + ".wav")
    this_sr, this_wav = siowav.read(wav_path)  # just for get this_sr
    if this_sr != sr:
        raise ValueError
    this_wav = np.float32(this_wav) / 32768.
    stftm = get_stftm(this_wav, sr, frame_shift, frame_size, n_fft, window)
    mel = get_mel(stftm, mel_filterbank)
    log_mel = log_compress(mel, floor_gate)     # [-1, 1]
    return log_mel

def get_key_lst(root):
    ret = []
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isfile(item_path):
            if item_path.endswith(".wav"):
                key = item[:-4]
                ret.append(key)
    return ret


def main():
    print(__file__)
    args = get_arguments()

    # 1st. get key_lst, which contains all the wav file keys.
    key_lst = get_key_lst(args.src_dir)
    assert key_lst, "[E] key list is empty!"

    # 2st. get mel-filterbank
    mel_filterbank = get_mel_filterbank(sr=args.sr, n_fft=args.n_fft, n_mels=args.n_mels)

    if not os.path.isdir(args.dest_dir):
        os.makedirs(args.dest_dir)
    for key in tqdm.tqdm(key_lst):
        try:
            np_log_mel = wav2mel(key=key, root=args.src_dir, sr=args.sr,
                                 frame_shift=args.frame_shift, frame_size=args.frame_size,
                                 n_fft=args.n_fft, window=args.window,
                                 mel_filterbank=mel_filterbank, floor_gate=args.floor_gate)
            np_log_mel.astype(np.float32).tofile(os.path.join(args.dest_dir, key + ".bin"))
        except ValueError as e:
            print(e, key)

    print("Congratulations!")


if __name__ == "__main__":
    main()

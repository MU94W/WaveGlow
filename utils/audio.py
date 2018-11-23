import tensorflow as tf
import numpy as np
from scipy import signal
import librosa


def mu_law(x, mu=255, int8=False, scope=None):
    """A TF implementation of Mu-Law encoding.
    Args:
        x: The audio samples to encode.
        mu: The Mu to use in our Mu-Law.
        int8: Use int8 encoding.
        scope:
    Returns:
        out: The Mu-Law encoded int8 data.
    """
    with tf.name_scope(scope, "mu_law", [x]):
        out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
        out = tf.floor(out * 128)
        if int8:
            out = tf.cast(out, tf.int8)
        return out


def inv_mu_law(x, mu=255, scope=None):
    """A TF implementation of inverse Mu-Law.
    Args:
        x: The Mu-Law samples to decode.
        mu: The Mu we used to encode these samples.
        scope:
    Returns:
        out: The decoded data.
    """
    with tf.name_scope(scope, "inv_mu_law", [x]):
        x = tf.cast(x, tf.float32)
        out = (x + 0.5) * 2. / (mu + 1)
        out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
        out = tf.where(tf.equal(x, 0), x, out)
        return out


def np_inv_mu_law(x, mu=255, scope=None):
    """A numpy implementation of inverse Mu-Law.
    Args:
        x: The Mu-Law samples to decode.
        mu: The Mu we used to encode these samples.
        scope:
    Returns:
        out: The decoded data.
    """
    x = np.float32(x)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out[x == 0.] = x[x == 0.]
    return out


def code_as_n_bits(x, bits=16, scope=None):
    with tf.name_scope(scope, "code_as_n_bits", [x]):
        coef = 1 << (bits-1)
        ceil = coef - 1
        floor = -coef
        x = tf.clip_by_value(x * coef, clip_value_max=ceil, clip_value_min=floor)
        return x


def preemphasis(x):
  return signal.lfilter([1, -0.97], [1], x)


def de_preemphasis(x):
  return signal.lfilter([1], [1, -0.97], x)


def get_stftm(wav, sr, frame_shift, frame_size, n_fft, window):
    tmp = np.abs(librosa.core.stft(y=wav, n_fft=n_fft, hop_length=int(frame_shift * sr),
                                   win_length=int(frame_size * sr), window=window))
    return tmp.T


def get_mel_filterbank(sr, n_fft, n_mels):
    # tmp = librosa.filters.mel(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    tmp = librosa.filters.mel(sr, n_fft, n_mels=n_mels)
    return tmp.T


def get_mel(stftm, mel_filterbank):
    return np.matmul(stftm, mel_filterbank)


def normalize(x, min_val, max_val):
    # return np.clip((x - min_val) / (max_val - min_val), 0., 1.)
    # 或许不该设置上限，另一方面，下限也已经由floor_gate保证了
    return (x - min_val) / (max_val - min_val)


def inv_normalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


tf_inv_normalize = inv_normalize


def log_compress(spec, floor_gate, min_db=-100., max_db=20.):
    return normalize(20 * np.log10(np.maximum(spec, floor_gate)), min_db, max_db)


def inv_log_compress(compressed_spec, min_db=-100., max_db=20.):
    return np.power(10., inv_normalize(compressed_spec, min_db, max_db) / 20.)


def tf_inv_log_compress(compressed_spec, min_db=-100., max_db=20.):
    return tf.pow(10., tf_inv_normalize(compressed_spec, min_db, max_db) / 20.)


def tf_griffin_lim(S, aug_by_power=1.2, frame_length=1200, frame_shift=300, n_fft=2048, sample_rate=24000, max_iter=50):
    with tf.name_scope('griffin_lim'):
        max_raw_batch = tf.reduce_max(tf.abs(S), axis=[1, 2], keep_dims=True)
        S = tf.pow(S, aug_by_power)
        max_init_batch = tf.reduce_max(tf.abs(S), axis=[1, 2], keep_dims=True)
        reverse_coef_batch = tf.div(max_raw_batch, max_init_batch)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = tf.contrib.signal.inverse_stft(S_complex, frame_length, frame_shift, n_fft)
        for _ in range(max_iter):
            est = tf.contrib.signal.stft(y, frame_length, frame_shift, n_fft, pad_end=False)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = tf.contrib.signal.inverse_stft(S_complex * angles, frame_length, frame_shift, n_fft)
        y *= reverse_coef_batch
        return y

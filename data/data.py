import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def parse_single_example(cond_dims):
    def __parse_single_example(example_proto):
        features = {"sr": tf.FixedLenFeature([], tf.int64),
                    "key_raw": tf.FixedLenFeature([], tf.string),
                    "wav_raw": tf.FixedLenFeature([], tf.string),   # uint16; little-endian
                    "frames": tf.FixedLenFeature([], tf.int64),
                    "cond": tf.FixedLenFeature([], tf.string)}
        parsed = tf.parse_single_example(example_proto, features=features)
        sr = tf.cast(parsed["sr"], tf.int32)
        key = parsed["key_raw"]
        wav = tf.reshape(tf.divide(tf.cast(tf.decode_raw(parsed["wav_raw"], tf.int16), dtype=tf.float32), 32768), (-1, 1))  # shape := (width, 1)
        frames = tf.cast(parsed["frames"], tf.int32)
        cond = tf.reshape(tf.decode_raw(parsed["cond"], tf.float32), (frames, cond_dims))
        return {"sr": sr, "key": key, "wav": wav, "cond": cond}
    return __parse_single_example


def crop_cond_wav(crop_frames, hop_length):
    def __crop(inputs):
        cond_frames = tf.shape(inputs["cond"])[0]
        check = control_flow_ops.Assert(
            math_ops.reduce_all(cond_frames >= crop_frames),
            ["Need value.shape >= size, got ", cond_frames, crop_frames],
            summarize=1000)
        dummy_crop_frames = crop_frames + 1  # since the last frame is calculated by pad.
        cond_limit = cond_frames - dummy_crop_frames + 1
        cond_random_offset = control_flow_ops.cond(tf.equal(cond_limit, 0),
                                                  lambda: 0,
                                                  lambda: tf.random_uniform(shape=(),
                                                                            maxval=tf.int32.max,
                                                                            dtype=tf.int32) % cond_limit)
        cond_random_offset = control_flow_ops.with_dependencies([check], cond_random_offset)
        wav_random_offset = cond_random_offset * hop_length
        crop_samples = (crop_frames - 1) * hop_length + 1
        crop_cond = tf.slice(inputs["cond"], [cond_random_offset, 0], [crop_frames, -1])  # use real crop frames
        crop_wav = tf.slice(inputs["wav"], [wav_random_offset, 0], [crop_samples, -1])
        return {"sr": inputs["sr"], "key": inputs["key"], "wav": crop_wav, "cond": crop_cond}
    return __crop


def filter_predictor(crop_frames, hop_length):
    def __predictor(inputs):
        return math_ops.logical_and(math_ops.greater_equal(tf.shape(inputs["cond"])[0], crop_frames),
                                    math_ops.greater_equal(tf.shape(inputs["wav"])[0], (crop_frames-1)*hop_length+1))
    return __predictor


def get_dataset(tfrecord_path, batch_size=16, crop_frames=40, hop_length=200, cond_dims=80, train_phase=True, shuffle_buffer_size=2000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example(cond_dims))
    if train_phase:
        dataset = dataset.filter(filter_predictor(crop_frames, hop_length))
        dataset = dataset.map(crop_cond_wav(crop_frames, hop_length))
        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(1)
    return dataset

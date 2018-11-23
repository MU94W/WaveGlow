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
        # NOTE: little-endian
        label_f, label_c = tf.split(tf.reshape(tf.decode_raw(parsed["wav_raw"], tf.uint8), (-1, 2)), 2, axis=-1)  # label_c shape := (width, 1)
        label_f, label_c = tf.cast(label_f, tf.int32), tf.cast(label_c, tf.int32)
        label_c = tf.where(tf.less(label_c, 128),
                            x=label_c,  # pos [0, 127]
                            y=label_c - 256)    # neg [128, 255] -> [-128, -1]
        label_c = label_c + 128     # [0, 255]
        wav = tf.reshape(tf.divide(tf.cast(tf.decode_raw(parsed["wav_raw"], tf.int16), dtype=tf.float32), 32768), (-1, 1))  # shape := (width, 1)
        frames = tf.cast(parsed["frames"], tf.int32)
        cond = tf.reshape(tf.decode_raw(parsed["cond"], tf.float32), (frames, cond_dims))
        return {"sr": sr, "key": key, "wav": wav, "cond": cond, "label_c": label_c, "label_f": label_f}
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
        crop_label_c = tf.slice(inputs["label_c"], [wav_random_offset, 0], [crop_samples, -1])
        crop_label_f = tf.slice(inputs["label_f"], [wav_random_offset, 0], [crop_samples, -1])
        return {"sr": inputs["sr"], "key": inputs["key"], "wav": crop_wav, "cond": crop_cond, "label_c": crop_label_c, "label_f": crop_label_f}
    return __crop


def filter_predictor(crop_frames, hop_length):
    def __predictor(inputs):
        return math_ops.logical_and(math_ops.greater_equal(tf.shape(inputs["cond"])[0], crop_frames),
                                    math_ops.greater_equal(tf.shape(inputs["wav"])[0], (crop_frames-1)*hop_length+1))
    return __predictor


def get_dataset(tfrecord_path, batch_size=16, crop_frames=40, hop_length=200, cond_dims=80, train_phase=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_single_example(cond_dims))
    if train_phase:
        dataset = dataset.filter(filter_predictor(crop_frames, hop_length))
        dataset = dataset.map(crop_cond_wav(crop_frames, hop_length))
        dataset = dataset.shuffle(10000).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(1)
    return dataset

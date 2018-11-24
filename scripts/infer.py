import argparse
import tensorflow as tf
import os
import tqdm
import json
import scipy.io.wavfile as siowav
import numpy as np
from models.WaveGlow.nnet.glow import WaveGlow as Config
from models.WaveGlow.data.data import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MOVING_AVERAGE_DECAY = 0.9999

def get_args():
    parser = argparse.ArgumentParser(description="Train WaveGlow!")
    parser.add_argument("--data_path", type=str, default="./assets/tfrecords")
    parser.add_argument("--log_dir", type=str, default="./assets/logs-wn")
    parser.add_argument("--hp_path", type=str, default="./hyperparams/config_16k.json")
    parser.add_argument("--sigma", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = get_args()
    # Parse configs.  Globals nicer in this case
    with open(args.hp_path) as f:
        hp = json.loads(f.read())
    net = Config(**hp["waveglow_config"])
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope("data"):
            dataset = get_dataset(args.data_path, 1, train_phase=False)
            iterator = dataset.make_one_shot_iterator()
            inputs = iterator.get_next()

        global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)
        net_tensor_dic = net.infer(inputs=inputs, sigma=args.sigma)


        # get shadow trainable variables.
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        trainable_var_lst = tf.trainable_variables()
        shadow_var_name_lst = [ema.average_name(var=item) for item in trainable_var_lst]

        all_var_dic = dict(zip(shadow_var_name_lst, trainable_var_lst))
        all_var_dic.update({"global_step": global_step})

        # get saver.
        saver = tf.train.Saver(var_list=all_var_dic)
        saver = tf.train.Saver()

    config = tf.ConfigProto(device_count = {"CPU": 12},
                            inter_op_parallelism_threads=6,)
                            #intra_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Session(graph=graph, config=config) as sess:
        # get checkpoint
        save_dir = os.path.join(args.log_dir, "save")
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        else:
            print("[E] No model found!")
            exit(1)

        global_step_eval = sess.run(global_step)
        try:
            gen_root = os.path.join(args.log_dir, "gen", "{}".format(global_step_eval))
            if not os.path.exists(gen_root) or not os.path.isdir(gen_root):
                os.makedirs(gen_root)
            while True:
                x, key = sess.run([net_tensor_dic["x"], inputs["key"]])
                key = os.path.basename(key[0].decode('utf-8'))
                path = os.path.join(gen_root, key + ".wav")
                siowav.write(path, rate=hp["sampling_rate"], data=np.int16(x[0] * 32768.))
                print("Synthesize {}".format(key))
        except tf.errors.OutOfRangeError:
            print("End of dataset")

    print("Congratulations!")


if __name__ == "__main__":
    main()

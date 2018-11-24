import argparse
import tensorflow as tf
import os
import tqdm
import json
from models.WaveGlow.nnet.glow import WaveGlow as Config
from models.WaveGlow.data.data import get_dataset
from models.WaveGlow.utils.multi_gpu import average_gradients

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LEARNING_RATE = 1e-4
MOVING_AVERAGE_DECAY = 0.9999
NUM_GPUS_DEFAULT = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def get_args():
    parser = argparse.ArgumentParser(description="Train WaveGlow!")
    parser.add_argument("--data_dir", type=str, default="./assets/tfrecords")
    parser.add_argument("--log_dir", type=str, default="./assets/logs")
    parser.add_argument("--hp_path", type=str, default="./hyperparams/config_16k.json")
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--batch_size", type=int, default=2*NUM_GPUS_DEFAULT)
    parser.add_argument("--save_per_steps", type=int, default=10000)
    parser.add_argument("--dev_per_steps", type=int, default=100)
    # multi-gpu config
    parser.add_argument("--num_gpus", type=int, default=NUM_GPUS_DEFAULT)
    return parser.parse_args()


def main():
    args = get_args()
    # Parse configs.  Globals nicer in this case
    with open(args.hp_path) as f:
        hp = json.loads(f.read())
    net = Config(**hp["waveglow_config"])
    graph = tf.Graph()
    with graph.as_default(), tf.device("/cpu:0"):
        batch_size_per_gpu = args.batch_size // args.num_gpus
        with tf.variable_scope("data"):
            train_set = get_dataset(os.path.join(args.data_dir, "train.tfrecords"), batch_size_per_gpu, **hp["data_config"])
            dev_set = get_dataset(os.path.join(args.data_dir, "dev.tfrecords"), batch_size_per_gpu, **hp["data_config"])
            train_iterator = train_set.make_one_shot_iterator()
            dev_iterator = dev_set.make_one_shot_iterator()

        # get optimizer.
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(LEARNING_RATE)

        tower_grads = []
        train_total_loss = 0.
        dev_total_loss = 0.
        for idx in range(args.num_gpus):
            with tf.device("/gpu:{}".format(idx)):
                with tf.name_scope("Tower_{}".format(idx)):
                    # build net.
                    # Note that iterator.get_next must be called for each sub graph.
                    train_net_tensor_dic = net(inputs=train_iterator.get_next(), reuse=tf.AUTO_REUSE)
                    dev_net_tensor_dic = net(inputs=dev_iterator.get_next(), reuse=tf.AUTO_REUSE)

                    # Reuse variables for the next tower.
                    # Only work if a variable_scope is specified around the for loop.
                    # tf.get_variable_scope().reuse_variables()

                    train_loss = train_net_tensor_dic["loss"]
                    train_total_loss += train_loss
                    dev_loss = dev_net_tensor_dic["loss"]
                    dev_total_loss += dev_loss
                    grads = opt.compute_gradients(train_loss)
                    tower_grads.append(grads)

        # Add loss summary.
        train_loss_summary = tf.summary.scalar("train/loss", train_total_loss / args.num_gpus)
        dev_loss_summary = tf.summary.scalar("dev/loss", dev_total_loss / args.num_gpus)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)

        trainable_vars = tf.trainable_variables()
        with tf.control_dependencies([apply_gradient_op]):
            upd = ema.apply(trainable_vars)

        # get saver.
        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # use all the RAM of GPUs! ;P
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # get checkpoint
        save_dir = os.path.join(args.log_dir, "save")
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        else:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, "log"), graph=graph)
        save_path = os.path.join(save_dir, net.name)

        global_step_eval = sess.run(global_step)
        pbar = tqdm.tqdm(total=args.steps)
        pbar.update(global_step_eval)
        summary_per_step = tf.summary.merge([train_loss_summary])
        while global_step_eval < args.steps:
            eval_lst = sess.run([summary_per_step, global_step, upd])
            global_step_eval = eval_lst[-2]
            summary_writer.add_summary(eval_lst[0], global_step=global_step_eval)
            if global_step_eval % args.dev_per_steps == 0:
                dev_eval_lst = sess.run([dev_loss_summary])
                summary_writer.add_summary(dev_eval_lst[0], global_step=global_step_eval)
            if global_step_eval % args.save_per_steps == 0:
                if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                saver.save(sess=sess, save_path=save_path, global_step=global_step_eval)
            pbar.update(1)
        summary_writer.close()

    print("Congratulations!")


if __name__ == "__main__":
    main()

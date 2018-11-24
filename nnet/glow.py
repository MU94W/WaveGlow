import tensorflow as tf
from scipy.stats import ortho_group
import numpy as np

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope


def fused_add_tanh_sigmoid_multiply(input_a, input_b, name=None):
  with tf.name_scope(name, "fused_add_tanh_sigmoid_multiply", [input_a, input_b]) as scope:
    with jit_scope():
      in_act = input_a + input_b
      t_logits, s_logits = tf.split(in_act, num_or_size_splits=2, axis=1)
      acts = tf.nn.tanh(t_logits) * tf.nn.sigmoid(s_logits)
      return acts


def wave_glow_loss(sigma=1.0):
  def _loss(z, log_s_list, log_det_W_list):
    for i, log_s in enumerate(log_s_list):
      if i == 0:
        log_s_total = tf.reduce_sum(log_s)
        log_det_W_total = log_det_W_list[i]
      else:
        log_s_total = log_s_total + tf.reduce_sum(log_s)
        log_det_W_total += log_det_W_list[i]
    
    loss = tf.reduce_sum(z*z) / (2*sigma*sigma) - log_s_total - log_det_W_total
    return loss / tf.cast(tf.size(z), tf.float32)
  return _loss


def invertible1x1conv(z, reverse=False, name="invertible1x1conv", reuse=tf.AUTO_REUSE):
  """
  Inputs:
    z: 3D Tensor, shape := [batch_size, group_size, n_of_groups]
  """
  with tf.variable_scope(name, reuse=reuse):
    n_channels = z.shape[1].value
    kernel_data_init = ortho_group.rvs(n_channels)
    if np.linalg.det(kernel_data_init) < 0.:
      kernel_data_init[:, 0] = -1. * kernel_data_init[:, 0]
    kernel_data_init = np.expand_dims(kernel_data_init, axis=0)
    kernel = tf.get_variable(name="kernel", shape=(1, n_channels, n_channels), initializer=tf.initializers.constant(kernel_data_init), dtype=tf.float32)
    if reverse:
      kernel_inv = tf.linalg.inv(kernel)
      z = tf.nn.conv1d(z, kernel_inv, stride=1, padding="VALID", data_format="NCW")
      return z
    else:
      batch_size, n_of_groups = tf.shape(z)[0], tf.shape(z)[2]
      log_det_W = tf.cast(batch_size, tf.float32) * tf.cast(n_of_groups, tf.float32) * tf.log(tf.linalg.det(tf.squeeze(kernel, axis=0)))
      z = tf.nn.conv1d(z, kernel, stride=1, padding="VALID", data_format="NCW")
      return z, log_det_W


def wavenet_block(x, c, n_layers, n_channels, kernel_size, name="wavenet_block", reuse=tf.AUTO_REUSE):
  """
  Inputs:
    x: 3D Tensor, shape := [batch_size, group_size, n_of_groups]
    c: 3D Tensor, shape := [batch_size, group_size, n_of_groups]
  """
  with tf.variable_scope(name, reuse=reuse):
    n_in_channels = x.shape[1].value
    n_cond_channels = c.shape[1].value
    # 1st. input layer
    init_out = tf.layers.conv1d(x, filters=n_channels, kernel_size=1, data_format="channels_first")
    # 2nd. wavenet-like layers
    last_out = init_out
    for idx in range(n_layers):
      dilation_rate = 2 ** idx
      conv_out = tf.layers.conv1d(last_out, filters=2*n_channels, kernel_size=kernel_size, padding="same", data_format="channels_first", dilation_rate=dilation_rate)
      cond_out = tf.layers.conv1d(c, filters=2*n_channels, kernel_size=1, use_bias=False, data_format="channels_first")
      acts = fused_add_tanh_sigmoid_multiply(conv_out, cond_out)

      if idx != n_layers - 1:
        res_skip_acts = tf.layers.conv1d(acts, filters=2*n_channels, kernel_size=1, data_format="channels_first")
        res_out, skip_out = tf.split(res_skip_acts, num_or_size_splits=2, axis=1)
        last_out = last_out + res_out
      else:
        skip_out = tf.layers.conv1d(acts, filters=n_channels, kernel_size=1, data_format="channels_first")
      
      if idx == 0:
        output = skip_out
      else:
        output = output + skip_out
    # 3rd. output layer
    output = tf.layers.conv1d(output, filters=2*n_in_channels, kernel_size=1, data_format="channels_first",
                              kernel_initializer=tf.zeros_initializer(),
                              bias_initializer=tf.zeros_initializer())
    return output


class WaveGlow:
  def __init__(self, n_flows, n_group, n_early_every, n_early_size, WN_config,
               up_sample_channels, up_sample_win_length, up_sample_hop_length,
               name="wave_glow"):
      assert(n_group % 2 == 0)
      self.n_flows = n_flows
      self.n_group = n_group
      self.n_early_every = n_early_every
      self.n_early_size = n_early_size
      self.WN_config = WN_config
      self.up_sample_channels = up_sample_channels
      self.up_sample_win_length = up_sample_win_length
      self.up_sample_hop_length = up_sample_hop_length
      self.name = name
  
  def up_sample(self, cond, name=None):
    """
    Inputs:
      cond: 3D Tensor, shape := [batch_size, channels, width]
    """
    with tf.variable_scope(name, "upsample", [cond]):
      up_l_width = self.up_sample_win_length // 2
      up_r_width = self.up_sample_win_length - up_l_width
      cond = tf.expand_dims(cond, axis=2) # (b, c, 1, w)
      us_out = tf.layers.conv2d_transpose(cond, filters=self.up_sample_channels, kernel_size=[1, self.up_sample_win_length], strides=[1, self.up_sample_hop_length], data_format="channels_first")
      us_out = tf.squeeze(us_out, axis=2)
      us_out = us_out[:, :, up_l_width:-up_r_width]   # time_steps: up_sample_hop_length * (frames - 1)
      return us_out

  def __call__(self, inputs, reuse=tf.AUTO_REUSE):
    """
    Inputs:
      x: 2D Tensor, shape := [batch_size, time]
      c: 3D Tensor, shape := [batch_size, channels, frame]
    """
    with tf.variable_scope(self.name, reuse=reuse):
      x = inputs["wav"]
      c = tf.transpose(inputs["cond"], (0, 2, 1))
      us_cond = self.up_sample(c)
      batch_size, time_steps = tf.shape(x)[0], tf.shape(x)[1]
      us_frames = tf.shape(us_cond)[2]
      n_of_groups = tf.math.minimum(tf.div(time_steps, self.n_group),
                                    tf.div(us_frames, self.n_group))
      us_cond_group = tf.reshape(us_cond[:, :, :n_of_groups*self.n_group],
                                 [batch_size, self.up_sample_channels, n_of_groups, self.n_group])
      us_cond_group = tf.reshape(tf.transpose(us_cond_group, (0, 2, 1, 3)), [batch_size, n_of_groups, self.up_sample_channels*self.n_group])
      us_cond_group = tf.transpose(us_cond_group, (0, 2, 1))
      x_group = tf.reshape(x[:, :n_of_groups*self.n_group],
                           [batch_size, n_of_groups, self.n_group])
      x_group = tf.transpose(x_group, (0, 2, 1))

      z_list, log_s_list, log_det_W_list = [], [], []

      for idx in range(self.n_flows):
        with tf.variable_scope("flow_{}".format(idx)):
          if idx % self.n_early_every == 0 and idx > 0:
            z_list.append(x_group[:, :self.n_early_size])
            x_group = x_group[:, self.n_early_size:]
        
          x_group, log_det_W = invertible1x1conv(x_group, reverse=False)
          log_det_W_list.append(log_det_W)

          n_half = int(x_group.shape[1].value / 2)
          x_group_0 = x_group[:, :n_half]
          x_group_1 = x_group[:, n_half:]

          output = wavenet_block(x_group_0, us_cond_group, **self.WN_config)
          log_s = output[:, n_half:]
          b = output[:, :n_half]
          x_group_1 = tf.exp(log_s) * x_group_1 + b
          log_s_list.append(log_s)

          x_group = tf.concat([x_group_0, x_group_1], axis=1)

      z_list.append(x_group)
      z = tf.concat(z_list, axis=1)

      loss = wave_glow_loss()(z, log_s_list, log_det_W_list)
      return {"loss": loss}

  def infer(self, inputs, sigma=0.6, reuse=tf.AUTO_REUSE):
    """
    Inputs:
      c: 3D Tensor, shape := [batch_size, channels, frame]
    """
    with tf.variable_scope(self.name, reuse=reuse):
      c = tf.transpose(inputs["cond"], (0, 2, 1))
      us_cond = self.up_sample(c)
      batch_size, us_frames = tf.shape(us_cond)[0], tf.shape(us_cond)[2]
      n_of_groups = tf.div(us_frames, self.n_group)
      us_cond_group = tf.reshape(us_cond[:, :, :n_of_groups*self.n_group],
                                 [batch_size, self.up_sample_channels, n_of_groups, self.n_group])
      us_cond_group = tf.reshape(tf.transpose(us_cond_group, (0, 2, 1, 3)), [batch_size, n_of_groups, self.up_sample_channels*self.n_group])
      us_cond_group = tf.transpose(us_cond_group, (0, 2, 1))

      n_remaining_channels = self.n_group - (self.n_flows - 1) // self.n_early_every * self.n_early_size
      z_group = tf.random_normal(shape=(batch_size, n_remaining_channels, n_of_groups), stddev=sigma)
      x_group = z_group

      for idx in reversed(range(self.n_flows)):
        with tf.variable_scope("flow_{}".format(idx)):
          n_half = int(x_group.shape[1].value / 2)
          x_group_0 = x_group[:, :n_half]
          x_group_1 = x_group[:, n_half:]

          output = wavenet_block(x_group_0, us_cond_group, **self.WN_config)
          log_s = output[:, n_half:]
          b = output[:, :n_half]
          x_group_1 = (x_group_1 - b) / tf.exp(log_s)
          x_group = tf.concat([x_group_0, x_group_1], axis=1)

          x_group = invertible1x1conv(x_group, reverse=True)

          if idx % self.n_early_every == 0 and idx > 0:
            z = tf.random_normal(shape=(batch_size, self.n_early_size, n_of_groups), stddev=sigma)
            x_group = tf.concat([z, x_group], axis=1)
      x = tf.reshape(tf.transpose(x_group, (0, 2, 1)), [batch_size, n_of_groups*self.n_group])
      return {"x": x}

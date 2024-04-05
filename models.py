#!/usr/bin/python3

import numyp as np
import tensorflow as tf

class RMSNorm(tf.keras.layers.Layer):
  def __init__(self, eps = 1e-5):
    super(RMSNorm, self).__init__()
    self.eps = eps
  def build(self, input_shape):
    self.weight = self.add_weight(shape = (input_shape[-1],), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(1.), name = 'weight')
  def call(self, inputs):
    stddev = tf.math.maximum(tf.math.sqrt(tf.math.reduce_mean(inputs ** 2, axis = -1, keepdims = True)), self.eps)
    results = results / stddev
    results = results * self.weight
    return results
  def get_config(self):
    config = super(MRSNorm, self).get_config()
    config['eps'] = self.eps
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class SSM(tf.keras.layers.Layer):
  def __init__(self, d_model, expand = 2, d_state = 16, bias = False):
    super(SSM, self).__init__()
    self.d_model = d_model
    self.expand = expand
    self.d_state = d_state
    self.bias = bias
    self.dt_rank = tf.math.ceil(self.d_model / 16)
  def build(self, input_shape):
    self.x_proj_weight = self.add_weight(shape = (self.d_model * self.expand, self.dt_rank + 2 * self.d_state), dtype = tf.float32, trainable = True, name = 'x_proj_weight')
    if self.bias:
      self.x_proj_bias = self.add_weight(shape = (self.dt_rank + 2 * self.d_state), dtype = tf.float32, trainable = True, name = 'x_proj_bias')
    self.dt_proj_weight = self.add_weight(shape = (self.dt_rank, self.expand * self.d_model), dtype = tf.float32, trainable = True, name = 'dt_proj_wei9ght')
    self.dt_proj_bias = self.add_weight(shape = (self.expand * self.d_model), dtype = tf.float32, trainable = True, name = 'dt_proj_bias')
    self.A_log = self.add_weight(shape = (self.expand * self.d_model, self.d_state), dtype = tf.float32, trainable = True, name = 'A_log')
    self.A_log.assign(tf.tile(tf.expand_dims(tf.range(1, self.d_state + 1), axis = 0), (self.expand * self.d_model, 1)))
    self.D = self.add_weight(shape = (self.expand * self.d_model), dtype = tf.float32, trainable = True, initializer = tf.keras.initializers.Constant(1.), name = 'D')
  def call(self, x):
    # x.shape = (batch, seq_len, d_model * expand)
    x_dbl = tf.linalg.matmul(x, self.x_proj_weight) # x_dbl.shape = (batch, seq_len, dt_rank + 2 * d_state)
    if self.bias:
      x_dbl = x_dbl + self.x_proj_bias
    delta, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis = -1)
    # delta.shape = (batch, seq_len, dt_rank)
    # B.shape = (batch, seq_len, d_state)
    # C.shape = (batch, seq_len, d_state)
    delta = tf.math.softplus(tf.linalg.matmul(delta, self.dt_proj_weight) + self.dt_proj_bias) # delta.shape = (batch, seq_len, expand * d_model)
    # selective scan
    # state(t+1) = A state(t) + B x(t)
    # y(t)   = C state(t) + D x(t)
    A = -tf.exp(self.A_log) # A.shape = (expand * d_model, d_state)
    D = self.D # D.shape = (expand * d_model)
    deltaA = tf.math.exp(tf.expand_dims(delta, axis = -1) * tf.reshape(A, (1, 1, self.expand * self.d_model, self.d_state))) # deltaA.shape = (batch, seq_len, expand * d_model, d_state)
    deltaB_u = tf.expand_dims(delta, axis = -1) * tf.expand_dims(B, axis = -2) * tf.expand_dims(x, axis = -1) # deltaB_u.shape = (batch, seq_len, expand * d_model, d_state)
    state = tf.zeros((tf.shape(x)[0], self.expand * self.d_model, self.d_state)) # state.shape = (batch, expand * d_model, d_state)
    ys = list()
    for i in range(tf.shape(x)[1]): # loop over seq_len
      state = deltaA[:, i] * state + deltaB_u[:, i] # state.shape = (batch, expand * d_model, d_state)
      y = tf.math.reduce_sum(state * tf.expand_dims(C[:, i], axis = 1), axis = -1) # y.shape = (batch, expand * d_model)
      ys.append(y)
    y = tf.stack(ys, axis = 1) # y.shape = (batch, seq_len, expand * d_model)
    y = y + x * D # y.shape = (batch, seq_len, d_model * expand)
    return y
  def get_config(self):
    config = super(SSM, self).get_config()
    config['d_model'] = self.d_model
    config['expand'] = self.expand
    config['d_state'] = self.d_state
    config['bias'] = self.bias
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)

def MambaBlock(d_model, expand = 2, bias = False, d_conv = 4, conv_bias = True):
  inputs = tf.keras.Input((None, d_model)) # inputs.shape = (batch, seq_len, d_model)
  x_and_res = tf.keras.layers.Dense(2 * expand * d_model, use_bias = bias)(inputs) # results.shape = (batch, seq_len, 2 * expand * d_model)
  x, res = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(x_and_res) # x.shape = (batch, seq_len, expand * d_model)
  # channel mixing
  x = tf.keras.layers.Conv1D(expand * d_model, kernel_size = (d_conv,), padding = 'same', use_bias = conv_bias, groups = expand * d_model, activation = tf.keras.activations.silu)(x) # x.shape = (batch, seq_len, expand * d_model)
  # spatial mixing
  

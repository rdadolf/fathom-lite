#!/usr/bin/env python

from itertools import cycle

import tensorflow as tf

from ..nn import NeuralNetworkModel, default_runstep
from ..dataset import Dataset
from ..data_loader import DataLoader, Batcher

class ImagenetModel(NeuralNetworkModel):
  @property
  def inputs(self):
    return self.images

  @property
  def labels(self):
    return self._labels

  @property
  def outputs(self):
    return self.logits

  @property
  def loss(self):
    return self.loss_op

  @property
  def train(self):
    return self.train_op

  def build_inputs(self):
    with self.G.as_default():
      self.image_size = 224 # side of the square image
      self.channels = 3
      self.n_input = self.image_size * self.image_size * self.channels

      self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels])

  def build_labels(self):
    with self.G.as_default():
      self.n_classes = 1000 + 1 # background class
      self._labels = tf.placeholder(tf.int64, [None])

  def build_evaluation(self):
    """Evaluation metrics (e.g., accuracy)."""
    self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), self.labels) # TODO: off-by-one?
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

  def build_hyperparameters(self):
    with self.G.as_default():
      self.learning_rate = 0.001
      self.training_iters = 200000
      self.batch_size = 64
      self.display_step = 1

      self.dropout = 0.8 # Dropout, probability to keep units

    # TODO: can this not be a placeholder?
    self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

  def build_loss(self, logits, labels):
    with self.G.as_default():
      # Define loss
      # TODO: does this labels have unexpected state?
      self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return self.loss_op

  def build_train(self, total_loss):
    with self.G.as_default():
      opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

      # Compute and apply gradients.
      #self.train_op = opt.minimize(total_loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      self.train_op = opt.minimize(total_loss)

    return self.train_op

  def load_data(self):
    dl = DataLoader('/data')
    self.image_data = dl.load('imagenet-inputs')
    self.label_data = dl.load('imagenet-labels')
    # Grab the dataset from the internet, if necessary
    #self.num_batches_per_epoch = self.dataset.num_examples_per_epoch() / self.batch_size

  def run(self, runstep=default_runstep, n_steps=1):
    self.load_data()

    image_batcher = Batcher(self.image_data)
    label_batcher = Batcher(self.label_data)
    with self.G.as_default():
      for step in xrange(0,n_steps):
        batch_images = image_batcher.next_batch(self.batch_size)
        batch_labels = label_batcher.next_batch(self.batch_size)

        if not self.forward_only:
          _, loss_value, acc = runstep(
              self.session,
              [self.train, self.loss, self.accuracy],
              feed_dict={self.images: batch_images, self._labels: batch_labels, self.keep_prob: self.dropout},
          )

          if step % self.display_step == 0:
            print "Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss_value) + ", Training Accuracy= " + "{:.5f}".format(acc)
        else:
          _ = runstep(
              self.session,
              self.outputs,
              feed_dict={self.images: batch_images, self._labels: batch_labels, self.keep_prob: 1.},
          )

        step += 1

      #print "Testing Accuracy:", runstep(self.session, [self.accuracy], feed_dict={self.images: self.mnist.test.images[:256], self._labels: self.mnist.test.labels[:256], self.keep_prob: 1.})

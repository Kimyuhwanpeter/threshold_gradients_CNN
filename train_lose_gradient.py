# -*- coding: utf-8 -*-
from absl import flags
from random import random, shuffle

import tensorflow as tf
import numpy as np
import os
import sys
import contextlib
import timeit

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Training text path")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("te_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Testing text path")

flags.DEFINE_string("te_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Testing image path")

flags.DEFINE_integer("load_size", 266, "Original input size")

flags.DEFINE_integer("img_size", 256, "Model input size")

flags.DEFINE_integer("img_ch", 3, "Input channels")

flags.DEFINE_integer("batch_size", 32, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Training epochs")

flags.DEFINE_integer("num_classes", 60, "Number of classes")

flags.DEFINE_float("lr", 0.001, "Training learning rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Restored the checkpoint path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def te_input_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if lab_list == 74:
        lab_list = 72
        lab = lab_list - 16
    elif lab_list == 75:
        lab_list = 73
        lab = lab_list - 16
    elif lab_list == 76:
        lab_list = 74
        lab = lab_list - 16
    elif lab_list == 77:
        lab_list = 75
        lab = lab_list - 16
    else:
        lab = lab_list - 16

    return img, lab

def tr_input_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch])

    if lab_list == 74:
        lab_list = 72
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 75:
        lab_list = 73
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 76:
        lab_list = 74
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    elif lab_list == 77:
        lab_list = 75
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)
    else:
        lab_list = lab_list - 16
        lab = tf.one_hot(lab_list, FLAGS.num_classes)

    return img, lab

@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):
        
    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        total_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)
    grads = tape.gradient(total_loss, model.trainable_variables)
    print(grads[0]) # 출력은 간단하게 진행했는데...
    # 각 레이어에 존재하는 gradient를 reduce_mean 해서 대표값을 선정 한 후
    # 각 레이어마다의 gradient의 대표값들에 대해 0이 나온 비율을 따지자
    # 그래서 0이 나오는 값이 50 %를 넘겼을 경우 학습 정지
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def main():
    model = tf.keras.applications.MobileNetV2(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch), include_top=False, pooling="avg")
    regularizer = tf.keras.regularizers.l2(0.00005)
    
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    h = model.output
    h = tf.keras.layers.Dense(FLAGS.num_classes)(h)
    model = tf.keras.Model(inputs=model.input, outputs=h)
    model.summary()
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("=========================")
            print("Checkpoint are restored!!")
            print("=========================")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0, usecols=0)
        te_img = [FLAGS.te_img_path + img for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(te_input_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)
        
        for epoch in range(FLAGS.epochs):

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(tr_input_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):

                batch_images, batch_labels = next(tr_iter)
                #print("Constant folded execution:", timeit.timeit(lambda: next(tr_iter), number = 1), "s")
                loss = cal_loss(model, batch_images, batch_labels)
                #print(loss)

                

if __name__ == "__main__":
    main()
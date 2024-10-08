from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras as nn
import math
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, LEARNING_RATE, save_model_dir, save_every_n_epoch, NUM_CLASSES
from prepare_data import generate_datasets
from models import get_model


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


# def process_features(features, data_augmentation):
#     image_raw = features['image_raw'].numpy()
#     image_tensor_list = []
#     for image in image_raw:
#         image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
#         image_tensor_list.append(image_tensor)
#     images = tf.stack(image_tensor_list, axis=0)
#     labels = features['label'].numpy()

#     return images, labels


parser = argparse.ArgumentParser()
parser.add_argument("--idx", default=0, type=int)


if __name__ == '__main__':
    #gpus = tf.config.list_physical_devices('GPU')
    #if gpus:
    #    try:
    #        for gpu in gpus:
    #            tf.config.experimental.set_memory_growth(gpu, True)
    #        logical_gpus = tf.config.list_logical_devices('GPU')
    #        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #    except RuntimeError as e:
    #        print(e)

    args = parser.parse_args()

    # get the dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # create model
    feature = get_model(args.idx)
    #feature.load_weights(filepath="/home/dragon/store/lc/Basic_CNNs_TensorFlow2-master_BRCT/model_weight/model/model")
    feature.load_weights(filepath="/home/dragon/store/lc/Basic_CNNs_TensorFlow2-master_CYP2C9_2/model_weight/model2/modelepoch-440")
    for i in range(5):
        print(feature.layers[i])
        feature.layers[i].trainable = False
    print_model_summary(network=feature)
    with open(save_model_dir + "_loss.txt","w") as file:
        file.write("model:%d\n" % (args.idx))

    # model = tf.keras.Sequential([feature,
    #                             #  tf.keras.layers.GlobalAveragePooling2D(),
    #                              tf.keras.layers.Dropout(rate=0.5),
    #                              tf.keras.layers.Dense(1024, activation="relu"),
    #                              tf.keras.layers.Dropout(rate=0.5),
    #                              tf.keras.layers.Dense(NUM_CLASSES),
    #                              tf.keras.layers.Softmax()])
    model = feature
    print_model_summary(network=model)


    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #optimizer = nn.optimizers.Adam(learning_rate=LEARNING_RATE)
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(images, labels):
        images = images[0]
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=labels, y_pred=predictions)

    # @tf.function
    def valid_step(images, labels):
        images = images[0]
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=labels, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        step = 0
        for images, labels in train_dataset:
            step += 1
            # images, labels = process_features(features, data_augmentation=True)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))

        for valid_images, valid_labels in valid_dataset:
            # valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                  EPOCHS,
                                                                  train_loss.result().numpy(),
                                                                  train_accuracy.result().numpy(),
                                                                  valid_loss.result().numpy(),
                                                                  valid_accuracy.result().numpy()))
        with open(save_model_dir + "_loss.txt","a") as file:
            file.write("%d\t%f\t%f\t%f\t%f\n" % (epoch, train_loss.result().numpy(), train_accuracy.result().numpy(), valid_loss.result().numpy(), valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    # save weights
    model.save_weights(filepath=save_model_dir, save_format='tf')

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)


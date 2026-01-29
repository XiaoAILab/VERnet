import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import tensorflow as tf
from configuration_e import save_model_dir
from prepare_data_e import generate_datasets
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--idx", default=0, type=int)

if __name__ == '__main__':
#    gpus = tf.config.list_physical_devices('GPU')
#    if gpus:
#        try:
#            for gpu in gpus:
#                tf.config.experimental.set_memory_growth(gpu, True)
#            logical_gpus = tf.config.list_logical_devices('GPU')
#            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#        except RuntimeError as e:
#            print(e)

    args = parser.parse_args()

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    #print("*** valid_dataset ***")
    #print(valid_dataset)
    #time.sleep(10)
    # load the model
    model = get_model(args.idx)
    model.load_weights(filepath=save_model_dir)
    # model = tf.saved_model.load(save_model_dir)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # @tf.function
    def test_step(images, labels):
        images = images[0]
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        print(labels)
        print(predictions)
        #with open("predictions_valid_model2_af3_af3.txt","a") as file:
        #    file.write(str(predictions))
        print(t_loss)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for test_image, test_labels in test_dataset:
        #test_images, test_labels = process_features(features, data_augmentation=False)
        test_step(test_image, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))

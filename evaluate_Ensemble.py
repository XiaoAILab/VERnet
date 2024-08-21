import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from configuration_e import save_model_dir
from prepare_data_e import generate_datasets
from models import get_model
import numpy as np
import scipy.io as scio

parser = argparse.ArgumentParser()
parser.add_argument("--idx", default=0, type=int)

if __name__ == '__main__':
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         print(e)

    args = parser.parse_args()

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    #model_list = ["model", "model11", "model12", "model21", "model22", "model31", "model32"]
    #model_list = ["model_1382_22_channel7_70_5/modelepoch-190", "model_1382_22_channel7_70_6/modelepoch-170", "model_1382_22_channel7_70_7/modelepoch-180"]
    model_list = ['model1/modelepoch-380', 'model2/modelepoch-440', 'model3/modelepoch-380']
    model_count = len(model_list)
    
    predictions_all = np.zeros((test_count,2,model_count))
    #labels_all = np.zeros(test_count)

    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for i in range(model_count):

        # load the models
        model = get_model(args.idx)
        model.load_weights(filepath=save_model_dir+model_list[i])
        # model = tf.saved_model.load(save_model_dir)

        # Get the accuracy on the test set
        #loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
        #test_loss = tf.keras.metrics.Mean()
        #test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # @tf.function
        def test_step(images, labels):
            images = images[0]
            predictions = model(images, training=False)
            with tf.compat.v1.Session() as sess:
                predictions_numpy = predictions.numpy()
            return predictions_numpy
            #labels_all = labels
            # predictions_all[:,:,0] = predictions_numpy
            # # t_loss = loss_object(labels, predictions)
            # print(labels)
            # # print(predictions)
            # test_loss(t_loss)
            # test_accuracy(labels, predictions)

        for test_image, test_labels in test_dataset:
            #test_images, test_labels = process_features(features, data_augmentation=False)
            predictions_now = test_step(test_image, test_labels)
            predictions_all[:,:,i] = predictions_now
            # print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
            #                                                 test_accuracy.result()))
    scio.savemat('predictions123.mat', {'A':predictions_all}) 
        #predictions_ensemble = predictions_all[:,:,0]
        #print(predictions_all)
        #predictions = tf.constant(predictions_all, dtype = tf.double)
    predictions_ensemble = np.copy(predictions_all[:,:,0])
    print(predictions_all)
    for i in range(1,model_count):
        for j in range(test_count):
            t1 = abs(predictions_ensemble[j,1] - 0.5)
            t2 = abs(predictions_all[j,1,i] - 0.5)
            #print(t1,t2)
            if(t1 < t2):
                predictions_ensemble[j,:] = predictions_all[j,:,i]
    #print(predictions_all[21,1,:])
    for j in range(test_count):
        if((predictions_ensemble[j,1] < 0.9) & (predictions_ensemble[j,1]> 0.1)):
        #if(1==0):
            vote = 0
            for i in range(model_count):
                if(predictions_all[j,1,i] > 0.5):
                    vote = vote + 1
            #print("%d:%d" % (j,vote))
            if(vote > 1):
                for i in range(model_count):
                    if(predictions_all[j,1,i] > predictions_ensemble[j,1]):
                        predictions_ensemble[j,:] = predictions_all[j,:,i]
            else:
                for i in range(model_count):
                    if(predictions_all[j,1,i] < predictions_ensemble[j,1]):
                        predictions_ensemble[j,:] = predictions_all[j,:,i]
    #print(predictions_ensemble)
    test_accuracy(test_labels, predictions_ensemble)
    #print(test_labels)
    #print(predictions_ensemble)
    for index in range(test_count):
        predictions_ensemble[index,0] = test_labels[index]
    print(predictions_ensemble)
    #scio.savemat('predictions_ensemble_test.mat', {'A':predictions_ensemble})
    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))

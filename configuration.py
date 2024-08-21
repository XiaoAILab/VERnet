
DEVICE = "gpu"   # cpu or gpu

# some training parameters
EPOCHS = 1000
BATCH_SIZE = 69
NUM_CLASSES = 2
IMAGE_HEIGHT = 377
IMAGE_WIDTH = 377
# 377
CHANNELS = 7
LEARNING_RATE = 1e-4

save_model_dir = "saved_model/model"
# save_model_dir = "model_weight/model_1382_22_channel7_70/modelepoch-400"
save_every_n_epoch = 10
test_image_dir = "dataset/test.tmp484/0/BRCA1_BRCT_V1000_h_nrint.mat"

#dataset_dir = "dataset/"
dataset_dir = "dataset/dataset1382_3_2_adj2/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.44660194
TEST_SET_RATIO = 0

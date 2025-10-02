import tensorflow as tf
# print(tf.__version__)
import numpy as np
import random
from matplotlib import pyplot as plt
# import cv2
import os
from efficientnet_like import model as efficientnet_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

tf.config.experimental.set_virtual_device_configuration(
    tf.config.list_physical_devices('GPU')[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]
)

train_data_x_1 = np.load('./MET_Dataset/select_image/painting_train_img_hori_55.npy')
train_data_x_2 = np.load('./MET_Dataset/select_image/engraving_train_img_hori_55.npy')
train_data_x_3 = np.load('./MET_Dataset/select_image/artifact_train_img_hori_55.npy')
train_data_x = np.concatenate((train_data_x_1, train_data_x_2, train_data_x_3), axis=0)
np.random.seed(1)
np.random.shuffle(train_data_x)
del train_data_x_1
del train_data_x_2
del train_data_x_3

train_x = [train_data_x[:, 0], train_data_x[:, 1]]
del train_data_x

train_y_1 = np.load('./MET_Dataset/select_image/painting_train_label_hori_55.npy')
train_y_2 = np.load('./MET_Dataset/select_image/engraving_train_label_hori_55.npy')
train_y_3 = np.load('./MET_Dataset/select_image/artifact_train_label_hori_55.npy')
train_y = np.concatenate((train_y_1, train_y_2, train_y_3), axis=0)
np.random.seed(1)
np.random.shuffle(train_y)

valid_data_x_1 = np.load('./MET_Dataset/select_image/painting_valid_img_hori_55.npy')
valid_data_x_2 = np.load('./MET_Dataset/select_image/engraving_valid_img_hori_55.npy')
valid_data_x_3 = np.load('./MET_Dataset/select_image/artifact_valid_img_hori_55.npy')
valid_data_x = np.concatenate((valid_data_x_1, valid_data_x_2, valid_data_x_3), axis=0)
np.random.seed(1)
np.random.shuffle(valid_data_x)
del valid_data_x_1
del valid_data_x_2
del valid_data_x_3

valid_x = [valid_data_x[:, 0], valid_data_x[:, 1]]
del valid_data_x

valid_y_1 = np.load('./MET_Dataset/select_image/painting_valid_label_hori_55.npy')
valid_y_2 = np.load('./MET_Dataset/select_image/engraving_valid_label_hori_55.npy')
valid_y_3 = np.load('./MET_Dataset/select_image/artifact_valid_label_hori_55.npy')
valid_y = np.concatenate((valid_y_1, valid_y_2, valid_y_3), axis=0)

# valid_y = np.load('./MET_Dataset/select_image/valid_label_hori_2gap_55.npy')
np.random.seed(1)
np.random.shuffle(valid_y)

test_data_x_1 = np.load('./MET_Dataset/select_image/painting_test_img_hori_55.npy')
test_data_x_2 = np.load('./MET_Dataset/select_image/engraving_test_img_hori_55.npy')
test_data_x_3 = np.load('./MET_Dataset/select_image/artifact_test_img_hori_55.npy')
test_data_x = np.concatenate((test_data_x_1, test_data_x_2, test_data_x_3), axis=0)
np.random.seed(1)
np.random.shuffle(test_data_x)
del test_data_x_1
del test_data_x_2
del test_data_x_3

test_x = [test_data_x[:, 0], test_data_x[:, 1]]
del test_data_x

test_y_1 = np.load('./MET_Dataset/select_image/painting_test_label_hori_55.npy')
test_y_2 = np.load('./MET_Dataset/select_image/engraving_test_label_hori_55.npy')
test_y_3 = np.load('./MET_Dataset/select_image/artifact_test_label_hori_55.npy')
test_y = np.concatenate((test_y_1, test_y_2, test_y_3), axis=0)
np.random.seed(1)
np.random.shuffle(test_y)


def combo_metrics(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred) * 0.8 + \
           tf.keras.metrics.binary_accuracy(y_true, y_pred) * 0.2

def combo_net(input_shape):
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                   input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.2, name="fen_dropout")(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.ReLU()(x)
    fen_model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    fragment1 = tf.keras.layers.Input(shape=input_shape, name='img1_input')
    fragment2 = tf.keras.layers.Input(shape=input_shape, name='img2_input')

    f1_feature = fen_model.call(fragment1)
    f2_feature = fen_model.call(fragment2)

    concatted_feature = tf.keras.layers.Concatenate()([f1_feature, f2_feature])
    # print(concatted_feature.shape)
    fc512 = tf.keras.layers.Dense(512)(concatted_feature)
    bn = tf.keras.layers.BatchNormalization()(fc512)
    relu = tf.keras.layers.ReLU()(bn)
    fc512 = tf.keras.layers.Dense(512)(relu)
    bn = tf.keras.layers.BatchNormalization()(fc512)
    relu = tf.keras.layers.ReLU()(bn)
    out1 = tf.keras.layers.Dense(1, activation='sigmoid', name='class_output')(relu)

    combo_net_model = tf.keras.models.Model([fragment1, fragment2], out1)
    # tf.compat.v1.Session().graph.finalize()
    return combo_net_model

# if not os.path.exists('hori_5_res50_1.h5'):
if not os.path.exists('hori_5_EfficientNetB0_25class_ft_1.h5'):
    model = combo_net([96, 96, 3])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5,
                                                                 decay_steps=3600,
                                                                 decay_rate=0.99,
                                                                 staircase=True)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-3,
    #                                                              decay_steps=2.4, decay_rate=0.99)
    lr = 0.0001
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule,  momentum=0.9, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[['binary_accuracy']])
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max',
                                                   verbose=1, patience=10, restore_best_weights=True)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cur_path + "VerificationCode/checkpoint/")
    # tf.keras.backend.clear_session()
    model.fit(x=train_x, y=train_y, batch_size=200, epochs=100,
              callbacks=[es_callback],
              validation_data=(valid_x, valid_y))

    del train_x
    del train_y
    del valid_x
    del valid_y

    results = model.evaluate(test_x, test_y)
    print("test loss, test acc:", results)

    # model.save('hori_5_res50_1.h5')
    model.save('hori_5_EfficientNetB0_25class_ft_1.h5')


else:
    # model = tf.keras.models.load_model('hori_5_res50_1.h5')
    model = tf.keras.models.load_model('hori_5_EfficientNetB0_25class_ft_1.h5')
    results = model.evaluate(test_x, test_y)
    print(model.summary())
    print("test loss, test acc:", results)

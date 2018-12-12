import pandas as pd
import os

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
import keras
from keras import utils as np_utils
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn import preprocessing

from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau


num_classes=5
IMG_WIDTH = 299
IMG_HEIGHT= 299
EPOCHS=1
BATCH_SIZE=32

DATA_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(DATA_PATH, '../yelp_photos', 'yelp_academic_dataset_photo.json')
MODEL_PATH = DATA_PATH + "/project_best_model.h5"


def get_data(df_in):
    labels = np.array(df_in.iloc[:, :1])
    img_id = np.array(df_in.iloc[:, 1:])
    img_id = np.ravel(img_id.reshape((1, len(img_id))))

    le = preprocessing.LabelEncoder()
    label_list = ['food', 'inside', 'outside', 'drink', 'menu']
    le.fit(label_list)
    y = le.transform(labels)

    images = []
    for i in img_id:
        img_path = os.path.join(DATA_PATH, '../yelp_photos', 'yelp_academic_dataset_photos', i) +'.jpg'
        image = imread(img_path)
        image_resized = resize(image, (IMG_HEIGHT, IMG_WIDTH))
        images.append(image_resized)

    x = np.array(images)
    return x, y

def build_model():

    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    inception_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    avg_glob_pool = GlobalAveragePooling2D(name='avg_glob_pool')(inception_model.output)
    fc_prediction = Dense(5, activation='softmax', name='fc_prediction')(avg_glob_pool)
    new_model = Model(inputs=inception_model.input, outputs=fc_prediction)
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    new_model.summary()

    return new_model

def call_back_list(weight_path):

    history = History()
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)

    # REDUCE_LR_CALLBACK = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                        patience=REDUCE_LR_PATIENCE, verbose=1, mode='min',
    #                                        min_delta=0.0001, cooldown=2, min_lr=1e-7)

    # EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
    #                                         patience=EARLY_STOPPING_PATIENCE)

    callbacks_list = [history, checkpoint]#, REDUCE_LR_CALLBACK, EARLY_STOPPING_CALLBACK]
    return callbacks_list



if __name__ == "__main__":

    #csv_file = pd.read_csv('../yelp_photos/yelp_academic_dataset_photo_features.csv')
    #print(csv_file.head())
    json_file = pd.read_json(TRAIN_DATA_PATH, orient='columns',lines=True)
    json_file.drop(['business_id', 'caption'], axis=1, inplace=True)
    #print(json_file.head())

    train_ids, valid_ids = train_test_split(json_file, test_size=100, train_size=100, stratify=json_file['label'])

    train_x, train_y = get_data(train_ids)
    valid_x, valid_y = get_data(valid_ids)

    train_y = np_utils.to_categorical(train_y)
    valid_y = np_utils.to_categorical(valid_y)

    print(train_y.shape)

    model = build_model()

    weight_path = "{}_weights.best.hdf5".format('project')

    callbacks_list = call_back_list(weight_path)

    history_log = [
        model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list,
                  validation_data=(valid_x, valid_y))]
    print('Saveing model...')

    model.load_weights(weight_path)
    model.save(MODEL_PATH)


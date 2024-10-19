import os

import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense
from keras.optimizers import Adam
from numpy.random import default_rng

train_dir = r'c:\prj\nalog-captcha'
pic_h = 100
pic_w = 200
channels = 1
len_captcha = 6
files_num = 5600

model_path = r'c:\prj\nalogru-captcha\Models'

captcha_list = []
characters = {}
for captcha in os.listdir(train_dir):
    captcha_list.append(captcha)

    captcha_code = captcha.split(".")[0]
    for i in captcha_code:
        characters[i] = characters.get(i, 0) + 1
symbols = list(characters.keys())
len_symbols = len(symbols)
print(f'Only {len_symbols} symbols have been used in images')


img_shape = (pic_h, pic_w, channels)

nSamples = len(captcha_list)  # the number of samples 'captchas'

X = np.zeros((nSamples, pic_h, pic_w, channels))
y = np.zeros((len_captcha, nSamples, len_symbols))

for i, captcha in enumerate(captcha_list):
    captcha_code = captcha.split('.')[0]
    captcha_cv2 = cv2.imread(os.path.join(train_dir, captcha), cv2.IMREAD_GRAYSCALE)
    captcha_cv2 = captcha_cv2 / 255.0
    captcha_cv2 = np.reshape(captcha_cv2, img_shape)
    targs = np.zeros((len_captcha, len_symbols))

    for a, b in enumerate(captcha_code):
        try:
            s = symbols.index(str(b))
            targs[a, s] = 1
        except:
            pass
    X[i] = captcha_cv2
    y[:, i] = targs

print("shape of X:", X.shape)
print("shape of y:", y.shape)

rng = default_rng(seed=1)
test_numbers = rng.choice(files_num, size=int(files_num * 0.15), replace=False)

X_test = X[test_numbers]
X_full = np.delete(X, test_numbers, 0)
y_test = y[:, test_numbers]
y_full = np.delete(y, test_numbers, 1)

val_numbers = rng.choice(int(files_num * 0.85), size=int(files_num * 0.15), replace=False)

X_val = X_full[val_numbers]
X_train = np.delete(X_full, val_numbers, 0)
y_val = y_full[:, val_numbers]
y_train = np.delete(y_full, val_numbers, 1)
print('Samples in train set:', X_train.shape[0])
print('Samples in test set:', X_test.shape[0])
print('Samples in validation set:', X_val.shape[0])

print("Подготовка ауг")
aug = iaa.Sequential([iaa.CropAndPad(
    px=((0, 10), (0, 35), (0, 10), (0, 35)),
    pad_mode=['edge'],
    pad_cval=1
), iaa.Rotate(rotate=(-8, 8))])

X_aug_train = None
y_aug_train = y_train
for i in range(20):
    X_aug = aug(images=X_train)
    if X_aug_train is not None:
        X_aug_train = np.concatenate([X_aug_train, X_aug], axis=0)
        y_aug_train = np.concatenate([y_aug_train, y_train], axis=1)
    else:
        X_aug_train = X_aug
print("Конец подготовка ауг")

captcha = Input(shape=(100, 200, channels))
x = Conv2D(32, (5, 5), padding='valid', activation='relu')(captcha)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
maxpool = MaxPooling2D((2, 2), padding='same')(x)
outputs = []
for i in range(6):
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(maxpool)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(len_symbols, activation='softmax', name=f'char_{i + 1}')(x)
    outputs.append(x)

model = Model(inputs=captcha, outputs=outputs)
model.summary(line_length=100)

reduce_lr = ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
earlystopping = EarlyStopping(monitor="val_loss",
                              mode="min", patience=10,
                              min_delta=1e-4,
                              restore_best_weights=True)

# Create a list of all the images and labels in the dataset
dataset, vocab, max_len = [], set(), 0

for file in os.listdir(train_dir):
    file_path = os.path.join(train_dir, file)
    label = os.path.splitext(file)[0]  # Get the file name without the extension
    dataset.append([file_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

history = model.fit(X_aug_train, [y_aug_train[i] for i in range(len_captcha)],
                    batch_size=16,
                    epochs=400,
                    verbose=1,
                    validation_data=(X_val, [y_val[i] for i in range(len_captcha)]),
                    callbacks=[earlystopping, reduce_lr])

try:
    model.save(model_path, overwrite=True)
except Exception as e:
    print(str(e))

new_model = tf.keras.models.load_model(model_path)
new_model.summary()
new_model.save(os.path.join(model_path, 'model.h5'))

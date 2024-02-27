import os, glob
import glob
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import keras.utils as image
from keras.datasets import mnist
from PIL import Image as pil_image

model = InceptionV3() # Load a model and its weights

def resize_mnist(x):
    x_list = []
    for i in range(x.shape[0]):
        img = image.array_to_img(x[i, :, :, :].reshape(28, 28, -1))
        #img.save("mnist-{0:03d}.png".format(i))
        img = img.resize(size=(299, 299), resample=pil_image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list)

def resize_do_nothing(x):
    return x

def inception_score(x, resizer, batch_size=32):
    r = None
    n_batch = (x.shape[0]+batch_size-1) // batch_size
    for j in range(n_batch):
        x_batch = resizer(x[j*batch_size:(j+1)*batch_size, :, :, :])
        r_batch = model.predict(preprocess_input(x_batch)) # r has the probabilities for all classes
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    p_y = np.mean(r, axis=0) # p(y)
    e = r*np.log(r/p_y) # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1) # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e) # Inception score

def image_inception_score(globfile):
    files = glob.glob(globfile)
    xs = None
    for f in files:
        img = image.load_img(f, target_size=(299, 299))
        x = image.img_to_array(img) # x.shape=(299, 299, 3)
        x = np.expand_dims(x, axis=0) # Add an axis of batch-size. x.shape=(1, 299, 299, 3)
        xs = x if xs is None else np.concatenate([xs, x], axis=0)
    return inception_score(xs, resize_do_nothing)

models = ["1-ToonYou", "2-Lyriel", "3-RcnzCartoon", "4-MajicMix", "5-RealisticVision", "6-Tusun", "7-FilmVelvia", "8-GhibliBackground"]

# for Imodel in models:
#     for pic in range(4):
#         print("Inception score ({}, {})".format(Imodel, pic), image_inception_score("splitted/{}/{}-*.gif".format(Imodel,pic)))

for Imodel in models:
    print("Inception score ({})".format(Imodel), image_inception_score("AnimateDiff/AnimateDiff/samples/{}/sample/*.gif".format(Imodel)))

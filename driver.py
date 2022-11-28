


import numpy as np
import cv2

from PIL import Image
from matplotlib import pyplot as plt
from keras.applications import imagenet_utils
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from animals import animals

while True:
    # filename = './nada.jpg' # no
    # filename = './frame2.jpg' # yes
    # filename = './clifford.jpg'
    filename = './bcat.jpg' # yes

    # frame = cv2.imread(filename)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('sample', img)

    key = cv2.waitKey(1) & 0xFF

    myimage = Image.open(filename)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    image_np = load_image_into_numpy_array(myimage)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # y1, x1, y2, x2
    left = tf.image.crop_and_resize(image=image_np_expanded,
                                    boxes=[[0, 0, 1, 0.55555]],
                                    box_indices=[0],
                                    crop_size=[224, 224])
    right = tf.image.crop_and_resize(image=image_np_expanded,
                                    boxes=[[0, 0.55555, 1, 1]],
                                    box_indices=[0],
                                    crop_size=[224, 224])

    # test = test.numpy()
    # print(type(test))
    #
    # plt.imshow(test[0] / 255.0)
    # plt.show()

    # test = left.numpy()[0] / 255.0
    # plt.imshow(test)
    # plt.show()
    mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

    preprocessed_image_l = preprocess_input(left)
    predictions_l = mobile.predict(preprocessed_image_l)
    results_l = imagenet_utils.decode_predictions(predictions_l)[0]

    preprocessed_image_r = preprocess_input(right)
    predictions_r = mobile.predict(preprocessed_image_r)
    results_r = imagenet_utils.decode_predictions(predictions_r)[0]
    # mobile = tf.keras.applications.mobilenet.MobileNet()



    # filter if animal and passes the threshold
    print("---LEFT---", results_l)
    animal_results_l = [(thing, probability) for tag, thing, probability in results_l if
                        (thing in animals and probability > 0.1)]
    print("Animal spotted:", animal_results_l) if animal_results_l else print("No animal spotted")

    print()

    print("---RIGHT---", results_r)
    animal_results_r = [(thing, probability) for tag, thing, probability in results_r if
                      (thing in animals and probability > 0.1)]
    print("Animal spotted:", animal_results_r) if animal_results_r else print("No animal spotted")

    print()
    print()
    print()
    print()

    if key == 27:
        cv2.destroyAllWindows()
        break




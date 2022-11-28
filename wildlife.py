

"""
# change arguments to look for flags to curtomize behavior of this script (sys.argv, sys.argv[0], sys.argv[1])



To start make a directory with this file and a file named inputs with the videos u want to process

error handle if inputs folder empty folder


bash script to...
- create virtual env
- autoinstall packages
- deactivate virtual env (optional?)



Directories (need a function to check if these directories already exit; if so, remove contents beofre and after)
build -> parsed images from the .AVI video
debug -> debug information including the specific image swhere animal was detected


# parse video

# convert to images in the build directory

# process images

# boolean if an animal is detected in

# delete images from build directory

"""

# extract images
import sys
import argparse
import cv2
print(cv2.__version__)
from os import path, mkdir, scandir  # path.isdir()


# analyze images
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

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

def setup_dir():
    build_dir_status = path.isdir('build')
    debug_dir_status = path.isdir('debug')

    if not build_dir_status:
        # make a directory
        print("build_dir_status", build_dir_status)
        mkdir('./build/')

    if not debug_dir_status:
        mkdir('./debug/')


def extract_images(vid_file):
    vid_name, vid_ext = vid_file[:vid_file.rfind('.')], vid_file[vid_file.rfind('.')+1:]

    try:
        mkdir("./build/" + vid_name)
        mkdir("./debug/" + vid_name)
    except FileExistsError:
        pass

    # print("vid_name", vid_name)

    count = 0
    vid_cap = cv2.VideoCapture("./inputs/" + vid_file)
    success, image = vid_cap.read()
    success = True  # ?
    while True:
        # vid_cap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, (count * 200))
        success, image = vid_cap.read()
        if success:
            count += 1
            print(f'New frame #{count}: ', success)
            cv2.imwrite("./build/%s/frame_%d.jpg" % (vid_name, count), image)  # save frame as JPEG file
        else:
            break
    return vid_name, count


def analyze_images(directory, num_images):
    """
    Process
    """
    print("analyze_images " + directory + '/' + "frame_" + str(num_images))

    def has_animal(image):  # returns a boolean with whether or not there is an animal
        pass



    animals_found = []  # this will have the paths of the frames w/ animals

    for img_num in range(1, num_images+1):
        filename = "./build/" + directory + "/frame_" + str(img_num) + ".jpg"
        print("beginning " + "./build/" + directory + "/frame_" + str(img_num) + ".jpg")

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


        # mobile = tf.keras.applications.mobilenet.MobileNet()

        preprocessed_image_l = preprocess_input(left)
        predictions_l = mobile.predict(preprocessed_image_l)
        results_l = imagenet_utils.decode_predictions(predictions_l)[0]

        preprocessed_image_r = preprocess_input(right)
        predictions_r = mobile.predict(preprocessed_image_r)
        results_r = imagenet_utils.decode_predictions(predictions_r)[0]

        # filter if animal and passes the threshold
        # print("---LEFT---", results_l)
        animal_results_l = [(thing, probability) for tag, thing, probability in results_l if
                            (thing in animals and probability > 0.3)]
        print("LEFT -- Animal spotted:", animal_results_l) if animal_results_l else print("No animal spotted")

        # print()

        # print("---RIGHT---", results_r)
        animal_results_r = [(thing, probability) for tag, thing, probability in results_r if
                            (thing in animals and probability > 0.3)]
        print("RIGHT -- Animal spotted:", animal_results_r) if animal_results_r else print("No animal spotted")

        if animal_results_l or animal_results_r:
            # if there is an animal...
            print("Animal detected in " + directory + "/frame_" + str(img_num) + ".jpg")
            animals_found.append(filename)

        print()
        print()

    if animals_found:
        for path in animals_found:
            # copy image to extracted file
            pass


if __name__=="__main__":
    # allow for customizations with flags
    # one can detaul the granularity of how many ms to get frames
    # (e.g. 1 frame per 1 second or 1 frame per 3 seconds)

    # print(sys.argv)

    # REMEMBER !
    # take the date created from the original video and modify the frames to
    # have the respective original video date as their "date created"

    # function to validate that there already is an inputs folder
    #       make sure folder ISNT empty
    #       validate that all files in it have video file formats that cv2.VideoCapture() accepts
    setup_dir()  # Automatically setup the main directories
    # for every video file in the inputs folder
    with scandir('./inputs/') as entries:
        for entry in entries:
            print("entry.name", entry.name)
            vid_name, num_images = extract_images(entry.name)
            analyze_images(vid_name, num_images)
    # extract_images(sys.argv[1])



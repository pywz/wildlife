

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
from os import path, mkdir, scandir, remove # path.isdir()


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

verbose = False
_all = False
store = False


def setup_dir(sub_folder):
    if not path.isdir('build'):
        # make a directory
        mkdir('./build/')

        if not path.isdir(f"build/{sub_folder}"):
            mkdir(f"build/{sub_folder}")

    pass


def extract_frames(vid_file, _path, sub_folder):
    vid_name, vid_ext = vid_file[:vid_file.rfind('.')], vid_file[vid_file.rfind('.')+1:]

    if not path.isdir(f"./build/{sub_folder}/"):
        mkdir(f"./build/{sub_folder}/")
    mkdir(f"./build/{sub_folder}/" + vid_name)
    # except FileExistsError:
    #     pass

    # print("vid_name", vid_name)

    count = 0
    vid_cap = cv2.VideoCapture(_path + vid_file)
    success, image = vid_cap.read()
    success = True  # ?
    while True:
        # vid_cap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, (count * 200))
        success, image = vid_cap.read()
        if success:
            count += 1
            if verbose:
                print(f'Extracted frame #{count}: ', success)
            cv2.imwrite(f"./build/{sub_folder}/{vid_name}/frame_{count}.jpg", image)  # save frame as JPEG file
        else:
            break
    if verbose:
        print(f"Extracted {count} frames from {vid_name}")
    return vid_name, count


def analyze_frames(vid_name, num_images, folder):
    """
    Process
    """
    # print("Analyzing " + directory + '/' + "frame_" + str(num_images))

    # def has_animal(image):  # returns a boolean with whether or not there is an animal
    #     pass

    animals_found = []  # this will have the paths of the frames w/ animals

    for img_num in range(1, num_images+1):
        filename = "./build/" + folder + "/" + vid_name + "/frame_" + str(img_num) + ".jpg"

        if verbose:
            print("Analyzing " + "./build/" + folder + "/" + vid_name + "/frame_" + str(img_num) + ".jpg")

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

        # print("---RIGHT---", results_r)
        animal_results_r = [(thing, probability) for tag, thing, probability in results_r if
                            (thing in animals and probability > 0.3)]

        if animal_results_l or animal_results_r:
            animals_found.append(filename)

        if verbose:
            print("LEFT -- Animal spotted:", animal_results_l) if animal_results_l else print("No animal spotted")
            print("RIGHT -- Animal spotted:", animal_results_r) if animal_results_r else print("No animal spotted")

            if animal_results_l or animal_results_r:
                # if there is an animal...
                print("Animal detected in " + "./build/" + folder + "/" + vid_name + "/frame_" + str(img_num) + ".jpg")

            print()

        if not store and filename not in animals_found:
            remove(filename)

        # TODO: edit the creation date of the current "filename" to match the video's creation date


if __name__=="__main__":
    # allow for customizations with flags
    # one can detaul the granularity of how many ms to get frames
    # (e.g. 1 frame per 1 second or 1 frame per 3 seconds)

    print(sys.argv)

    flags = [_ for _ in sys.argv if _[0] == '-']

    for cmd in sys.argv:
        if cmd[0] == '-':
            if cmd[1] == 'v':
                verbose = True
            elif cmd[1] == 'a':
                _all = True
            elif cmd[1] == 's':
                store = True
            else:
                raise NotImplementedError(f"Flag -{cmd[1]} has not been implemented")

    want = [_ for _ in sys.argv if (_[0] != '-' and _ != 'wildlife.py')]

    # $ python3 wildlife.py -a Beaman
    # $ python3 wildlife.py -a Beaman MillsCreek
    # if -a set, go to specified folder/s

    if _all:
        pass
    else:
        folder = want[0]  # Beaman

        sub_folders = want[1:]  # ["bm-y22-m01-s01-c20", "bm-y22-m01-s01-c21"]

        # validate that this folder exists
        if not path.exists('./' + folder):
            raise NotADirectoryError(f"{folder} is not a folder")

        # validate sub_folders in folder

        for sub in sub_folders:
            if not path.exists('./' + folder + '/' + sub):
                raise NotADirectoryError(f"{sub} is not a sub folder of {folder}")

        for sub_folder in sub_folders:
            cur_path = f"./{folder}/{sub_folder}/"
            with scandir(cur_path) as videos:
                for video in videos:
                    print(f"↓ Starting {cur_path}{video.name}")
                    setup_dir(sub_folder)
                    vid_name, num_images = extract_frames(video.name, cur_path, sub_folder)
                    analyze_frames(vid_name, num_images, sub_folder)
            # extract_images(sys.argv[1])





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


import sys
import argparse
import cv2
print(cv2.__version__)
from os import path, mkdir, scandir  # path.isdir()

from animals import animals


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
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, (count*1000))
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

    #
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



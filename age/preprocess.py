import math
import numpy as np
import pickle
import errno
import argparse
import os
import re
import csv
import cv2
import face_recognition
import imutils
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
from datetime import date, timedelta

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('-s', '--save-dir', default="preprocessed_data")
parser.add_argument('-r', '--regularize', action='store_true')
parser.add_argument(
    '-f',
    '--force',
    action='store_true',
    help='force re-regularize the image even if it already exists')
parser.add_argument('-ds',
                    '--datasets',
                    nargs='+',
                    default=['imdb-wiki', 'allagefaces', 'appa'])
args = parser.parse_args()


## A utility function ##########################################################
def align_and_crop(filename):
    image = face_recognition.load_image_file(filename)
    locations = face_recognition.face_locations(image, model="hog")
    landmarks = face_recognition.face_landmarks(image)

    if len(locations) == 0 or len(landmarks) == 0:
        return None

    left_eye_pts = landmarks[0]['left_eye']
    right_eye_pts = landmarks[0]['right_eye']

    left_eye_center = np.array(left_eye_pts).mean(axis=0).astype("int")
    right_eye_center = np.array(right_eye_pts).mean(axis=0).astype("int")
    left_eye_center = (left_eye_center[0], left_eye_center[1])
    right_eye_center = (right_eye_center[0], right_eye_center[1])

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    desired_left_eye = (0.39, 0.5)
    desired_right_eye_x = 1 - desired_left_eye[0]
    desired_image_width = 224
    desired_image_height = 224

    dist = np.sqrt((dx**2) + (dy**2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_image_width
    scale = desired_dist / dist

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                   (left_eye_center[1] + right_eye_center[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tx = desired_image_width * 0.5
    ty = desired_image_height * desired_left_eye[1]
    M[0, 2] += (tx - eyes_center[0])
    M[1, 2] += (ty - eyes_center[1])

    (y2, x2, y1, x1) = locations[0]
    output = cv2.warpAffine(image,
                            M, (desired_image_width, desired_image_height),
                            flags=cv2.INTER_CUBIC)

#    print("")
#    print(dy, dx)
#    print(angle)
#    cv2.imshow("image", image)
#    cv2.imshow("ouput", output)
#    while True:
#        key = cv2.waitKey(1) & 0xFF
#        if key == ord('q'):
#            break
#    exit(1)

    return output


## IMDB-WIKI DATASET ###########################################################
if 'imdb-wiki' in args.datasets:
    print(
        "IMDB-WIKI:"
    )  ## Parse data ##################################################################
    print("Loading Data...")
    imdbData = loadmat('./data/imdb_crop/imdb.mat')['imdb'][0][0]
    wikiData = loadmat('./data/wiki_crop/wiki.mat')['wiki'][0][0]

    # columns:
    # 0: dob: date of birth (Matlab serial date number)
    # 1: photo_taken: year when the photo was taken
    # 2: full_path: path to file
    # 3: gender: 0 for female and 1 for male, NaN if unknown
    # 4: name: name of the celebrity
    # 5: face_location: location of the face.
    # 6: face_score: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    # 8: celeb_names (IMDB only): list of all celebrity names
    # 9: celeb_id (IMDB only): index of celebrity name
    print("Parsing Data...")
    imdb = [(
        'data/imdb_crop/' + path[0],
        (date.fromisoformat('{}-07-01'.format(imdbData[1][0][i])) -
         (date.fromordinal(int(imdbData[0][0][i])) - timedelta(days=366))).days
        / 365.25) for i, path in enumerate(imdbData[2][0])
            if (not math.isinf(imdbData[6][0][i])) and (
                imdbData[0][0][i] > 366) and math.isnan(imdbData[7][0][i])]
    imdb = list(filter(lambda x: x[1] >= 0 and x[1] <= 100, imdb))
    imdbFeatures, imdbLabels = map(list, zip(*imdb))
    print("Parsed {} entries in the imdb dataset".format(len(imdb)))

    wiki = [(
        'data/wiki_crop/' + path[0],
        (date.fromisoformat('{}-07-01'.format(wikiData[1][0][i])) -
         (date.fromordinal(int(wikiData[0][0][i])) - timedelta(days=366))).days
        / 365.25) for i, path in enumerate(wikiData[2][0])
            if (not math.isinf(wikiData[6][0][i])) and (
                wikiData[0][0][i] > 366) and math.isnan(wikiData[7][0][i])]
    wiki = list(filter(lambda x: x[1] >= 0 and x[1] <= 100, wiki))
    wikiFeatures, wikiLabels = map(list, zip(*wiki))
    print("Parsed {} entries in the wiki dataset".format(len(wiki)))

    imdbWikiFeatures = np.concatenate([imdbFeatures, wikiFeatures])
    imdbWikiLabels = np.concatenate([imdbLabels, wikiLabels])

    ## Save data ###################################################################
    try:
        os.mkdir(args.save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print("Saving data...")
    np.save(args.save_dir + '/imdbWikiFeatures',
            imdbWikiFeatures,
            allow_pickle=False)
    np.save(args.save_dir + '/imdbWikiLabels',
            imdbWikiLabels,
            allow_pickle=False)

## ALL-AGE-FACE DATASET ########################################################
if 'allagefaces' in args.datasets:
    print("")
    print("ALL-AGE-FACE:")
    ## Parse data ##################################################################
    print("Loading Data...")
    allAgeFacesFiles = []
    with open('data/All-Age-Faces Dataset/image sets/train.txt') as f:
        allAgeFacesFiles = f.read().split('\n')
    with open('data/All-Age-Faces Dataset/image sets/val.txt') as f:
        allAgeFacesFiles += f.read().split('\n')

    if args.regularize:
        output_dir = 'data/All-Age-Faces Dataset/cropped'
        try:
            os.mkdir(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        entryPattern = re.compile(r'\d{5}A(\d{2})\.jpg (0|1)')
        allAgeFacesFeatures=[]
        allAgeFacesLabels=[]
        for i, f in enumerate(allAgeFacesFiles):
            if len(f.strip()) == 0:
                continue
            print('\rparsing {}/{}'.format(i, len(allAgeFacesFiles)), end="")

            # check if file exists
            dst_file_name = output_dir + '/' + f.split(' ')[0] + '_cropped.jpg'
            if os.path.exists(dst_file_name) and not args.force:
                allAgeFacesFeatures.append(dst_file_name)
                allAgeFacesLabels.append(int(entryPattern.match(f).groups()[0]))
                continue

            # parse file
            aligned_face = align_and_crop(
                'data/All-Age-Faces Dataset/original images/{}'.format(
                    f.split(' ')[0]))

            # save file
            if aligned_face is None:
                continue
            else:
                image = Image.fromarray(aligned_face)
                image.save(dst_file_name)
                allAgeFacesFeatures.append(dst_file_name)
                allAgeFacesLabels.append(int(entryPattern.match(f).groups()[0]))
    else:
        print("Parsing Data...")
        allAgeFacesFiles = list(filter(len, allAgeFacesFiles))
        allAgeFacesFiles = list(set(allAgeFacesFiles))
        print("Parsed {} entries in the all-age-face dataset".format(
            len(allAgeFacesFiles)))

        allAgeFacesFeatures = [
            'data/All-Age-Faces Dataset/aglined faces/' + e.split(' ')[0]
            for e in allAgeFacesFiles
        ]
        entryPattern = re.compile(r'\d{5}A(\d{2})\.jpg (0|1)')
        allAgeFacesLabels = [
            int(entryPattern.match(e).groups()[0]) for e in allAgeFacesFiles
        ]

    ## Save data ###################################################################
    print("Saving data...")
    np.save(args.save_dir + '/allAgeFacesFeatures',
            allAgeFacesFeatures,
            allow_pickle=False)
    np.save(args.save_dir + '/allAgeFacesLabels',
            allAgeFacesLabels,
            allow_pickle=False)

## APPA-REAL-FACE DATASET ######################################################
if 'appa' in args.datasets:
    print("")
    print("APPA-REAL-FACE:")

    ## Parse data ##################################################################
    appaFeatuers = []
    appaLabels = []
    print("Loading and Parsing Data...")
    for mode in ['train', 'test', 'valid']:
        print("parsing {}...".format(mode))
        with open('data/appa-real-release/gt_avg_{}.csv'.format(mode)) as f:
            reader = csv.DictReader(f)
            if args.regularize:
                for i, row in enumerate(reader):
                    print('\rparsing {}'.format(i), end="")

                    # check existing file
                    dst_file_name = 'data/appa-real-release/{}/{}'.format(
                        mode, row['file_name'] + '_cropped.jpg')
                    if os.path.exists(dst_file_name) and not args.force:
                        appaFeatuers.append(dst_file_name)
                        appaLabels.append(int(row['real_age']))
                        continue

                    # regularizefile
                    aligned_face = align_and_crop(
                        'data/appa-real-release/{}/{}'.format(
                            mode, row['file_name']))

                    # sace file
                    if aligned_face is None:
                        continue
                    else:
                        image = Image.fromarray(aligned_face)
                        image.save(dst_file_name)
                        appaFeatuers.append(dst_file_name)
                        appaLabels.append(int(row['real_age']))
            else:
                for i, row in enumerate(reader):
                    print('\rparsing {}'.format(i), end="")
                    appaFeatuers.append('data/appa-real-release/{}/{}'.format(
                        mode, row['file_name'] + '_face.jpg'))
                    appaLabels.append(int(row['real_age']))
    print("Parsed {} entries in the all-age-face dataset".format(
        len(appaFeatuers)))

    ## Save data ###################################################################
    print("Saving data...")
    np.save(args.save_dir + "/appaFeatuers", appaFeatuers, allow_pickle=False)
    np.save(args.save_dir + "/appaLabels", appaLabels, allow_pickle=False)

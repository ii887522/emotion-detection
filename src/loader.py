import numpy as np
import csv
from itertools import islice
import constants
from typing import Set
import cv2


"""
{
    "Training": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    },
    "PublicTest": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    },
    "PrivateTest": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    }
}
"""
def load_dataset(usages: Set[str] = set(["Training", "PublicTest", "PrivateTest"])) -> dict:
    input = {
        "Training": {},
        "PublicTest": {},
        "PrivateTest": {}
    }

    print("Loading dataset pixels...")
    pixels = load_dataset_pixels(usages)
    input["Training"]["x"] = pixels["Training"]["x"]
    input["PublicTest"]["x"] = pixels["PublicTest"]["x"]
    input["PrivateTest"]["x"] = pixels["PrivateTest"]["x"]

    print("Loading dataset labels...")
    labels = load_dataset_labels(usages)
    input["Training"]["y"] = labels["Training"]["y"]
    input["PublicTest"]["y"] = labels["PublicTest"]["y"]
    input["PrivateTest"]["y"] = labels["PrivateTest"]["y"]

    return input


"""
{
    "Training": {
        "x": (N, 48, 48, 1),
    },
    "PublicTest": {
        "x": (N, 48, 48, 1),
    },
    "PrivateTest": {
        "x": (N, 48, 48, 1),
    }
}
"""
def load_dataset_pixels(usages: Set[str] = set(["Training", "PublicTest", "PrivateTest"])) -> dict:
    input = {
        "Training": {},
        "PublicTest": {},
        "PrivateTest": {}
    }

    with open(f"{constants.EMOTION_DIR_PATH}/fer2013.csv") as old_fer_file:
        for row in islice(csv.reader(old_fer_file), 1, None):
            [_, pixels, usage] = row

            if usage not in usages:
                continue

            h_flipped_pixels = None

            if usage == "Training":
                # Reshape the pixels to fit OpenCV functions
                pixels = np.asarray(pixels.split(" "), np.uint8).reshape(constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1])

                # Data augmentation: Get the horizontally flipped pixels
                h_flipped_pixels = cv2.flip(pixels, 1)

                # Normalize the pixels
                pixels = pixels / 255.0
                h_flipped_pixels = h_flipped_pixels / 255.0

                # Reshape the pixels to fit CNN input layer
                pixels = np.expand_dims(pixels, 2)
                h_flipped_pixels = np.expand_dims(h_flipped_pixels, 2)

            else:
                # Reshape the pixels to fit CNN input layer
                pixels = np.asarray(pixels.split(" "), np.uint8).reshape(constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1], 1)

                # Normalize the pixels
                pixels = pixels / 255.0

            if input[usage].get("x"):
                input[usage]["x"].append(pixels)

            else:
                input[usage]["x"] = [pixels]

            if usage == "Training":
                input[usage]["x"].append(h_flipped_pixels)

        input["Training"]["x"] = np.array(input["Training"].get("x", []))
        input["PublicTest"]["x"] = np.array(input["PublicTest"].get("x", []))
        input["PrivateTest"]["x"] = np.array(input["PrivateTest"].get("x", []))

    return input


"""
{
    "Training": {
        "y": (N,)
    },
    "PublicTest": {
        "y": (N,)
    },
    "PrivateTest": {
        "y": (N,)
    }
}
"""
def load_dataset_labels(usages: Set[str] = set(["Training", "PublicTest", "PrivateTest"])) -> dict:
    input = {
        "Training": {},
        "PublicTest": {},
        "PrivateTest": {}
    }

    with open(f"{constants.EMOTION_DIR_PATH}/fer2013new.csv") as new_fer_file:
        for row in islice(csv.reader(new_fer_file), 1, None):
            [
                usage,
                _,
                neutral_vote_count,
                happiness_vote_count,
                surprise_vote_count,
                sadness_vote_count,
                anger_vote_count,
                disgust_vote_count,
                fear_vote_count,
                contempt_vote_count,
                unknown_vote_count,
                nf_vote_count
            ] = row

            if usage not in usages:
                continue

            vote_counts = [
                neutral_vote_count,
                happiness_vote_count,
                surprise_vote_count,
                sadness_vote_count,
                anger_vote_count,
                disgust_vote_count,
                fear_vote_count,
                contempt_vote_count,
                unknown_vote_count,
                nf_vote_count
            ]

            vote_count = vote_counts.index(max(vote_counts))

            if input[usage].get("y"):
                input[usage]["y"].append(vote_count)

            else:
                input[usage]["y"] = [vote_count]

            if usage == "Training":
                # Insert 1 more vote count because we have augmented 1 more image pixels
                input[usage]["y"].append(vote_count)

        input["Training"]["y"] = np.array(input["Training"].get("y", []))
        input["PublicTest"]["y"] = np.array(input["PublicTest"].get("y", []))
        input["PrivateTest"]["y"] = np.array(input["PrivateTest"].get("y", []))

    return input

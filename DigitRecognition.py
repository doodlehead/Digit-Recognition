import sys
import numpy as np
import pandas as pd
import cv2 as cv
from typing import List, Tuple
import argparse
import Levenshtein
from collections import Counter
from datetime import datetime

BIG_IMG_SIZE = 50

parser = argparse.ArgumentParser(description='Digit Recognizer')
parser.add_argument('-p', default=False, action='store_true', help='Run image preprocessing')
parser.add_argument('-c', default=False, action='store_true', help='Run freeman code computation')
parser.add_argument('-s', type=str, help='The file to run the prediction on')


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Apply processing steps an image in the np.ndarray format and return the result
    :param image: The image in np.ndarray format
    :return: A tuple containing the label and processed image
    """
    # Thresholding
    _, img_np = cv.threshold(image, 120, 255, cv.THRESH_BINARY)

    # Crop
    coords = cv.findNonZero(img_np)
    x, y, w, h = cv.boundingRect(coords)
    cx, cy = x + w // 2, y + h // 2
    min_size = max(h, w)
    cropped = img_np[y:min(27, y + min_size),
              max(0, cx - min_size // 2):min(27, cx + min_size // 2)]

    # Add a border
    border = cv.copyMakeBorder(cropped, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    # Normalize size
    resize = cv.resize(border, dsize=(BIG_IMG_SIZE, BIG_IMG_SIZE))

    return resize


def process_image_series(image_data: pd.Series) -> Tuple[int, np.ndarray]:
    """
    Apply processing steps an image in the pd.Series format and return the result
    :param image_data: A Series containing the label and the pixel data for a digit
    :return: A tuple containing the label and processed image
    """
    label = image_data[0]
    pixel_data = image_data[1:]

    # Convert to 2d image format
    img_np = pixel_data.to_numpy(dtype=np.uint8).reshape((28, 28))

    return label, process_image(img_np)


def preprocess_data(data: pd.DataFrame, output_name: str) -> None:
    """
    Preprocess the data and then saves it as a CSV file for later use
    :param data: The data to process
    :param output_name: The name of the file to save to
    :return: None
    """
    # load training data, each row is an image

    # Create column labels for the DataFrame
    column_labels = ['label']
    for i in range(BIG_IMG_SIZE * BIG_IMG_SIZE):
        column_labels.append(f'pixel{i}')

    row_list = []

    for index, row in data.iterrows():
        label, img = process_image_series(row)
        values = np.insert(img.flatten(), 0, label)

        new_row = dict(zip(column_labels, values))
        row_list.append(new_row)

    # Save as csv
    processed = pd.DataFrame(row_list, columns=column_labels)
    processed.to_csv(output_name)


def get_value(arr: np.ndarray, row: int, col: int) -> any:
    """
    Helper function to get a value from a 2D array returning 0 if out of bounds.
    :param arr: The array to access
    :param row: The row index
    :param col: The column index
    :return: The retrieved value at [row, column] or 0 if out of bounds
    """
    y, x = arr.shape
    if (row < 0 or row >= y) or (col < 0 or col >= x):
        return 0
    return arr[row, col]


def get_freeman_code(img: np.ndarray) -> str:
    """
    Computes the Freeman code for an image
    :param img: An image of a digit
    :return: The computed Freeman code as a string
    """
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    outline = np.zeros((BIG_IMG_SIZE, BIG_IMG_SIZE), dtype=np.uint8)
    cv.drawContours(outline, contours, -1, (255, 255, 255))

    start = (-1, -1)
    h, w = img.shape

    # Locate the start pixel
    for i in range(h):
        if outline[i, i] == 255:
            start = (i, i)
            break
        elif get_value(outline, i + 1, i) == 255:
            start = (i + 1, i)
            break
        elif get_value(outline, i, i + 1) == 255:
            start = (i, i + 1)
            break

    if start == (-1, -1):
        print('Something went wrong, cannot find edge')
        pass

    # Iterate over the edge
    iter_count = 0
    max_limit = 180
    cx, cy = start
    code = ''

    visited: set[tuple[int, int]] = set()

    while iter_count < max_limit:
        d = 1  # distance away to check
        found = False
        # Check in clockwise direction, according to freeman code directions
        while not found:
            found = True
            # Don't allow backtracking
            if get_value(outline, cy - d, cx) == 255 and (cy - d, cx) not in visited:
                code += '0'
                cy = cy - d
            elif get_value(outline, cy - d, cx + d) == 255 and (cy - d, cx + d) not in visited:
                code += '1'
                cy, cx = cy - d, cx + d
            elif get_value(outline, cy, cx + d) == 255 and (cy, cx + d) not in visited:
                code += '2'
                cx = cx + d
            elif get_value(outline, cy + d, cx + d) == 255 and (cy + d, cx + d) not in visited:
                code += '3'
                cy, cx = cy + d, cx + d
            elif get_value(outline, cy + d, cx) == 255 and (cy + d, cx) not in visited:
                code += '4'
                cy = cy + d
            elif get_value(outline, cy + d, cx - d) == 255 and (cy + d, cx - d) not in visited:
                code += '5'
                cy, cx = cy + d, cx - d
            elif get_value(outline, cy, cx - d) == 255 and (cy, cx - d) not in visited:
                code += '6'
                cx = cx - d
            elif get_value(outline, cy - d, cx - d) == 255 and (cy - d, cx - d) not in visited:
                code += '7'
                cy, cx = cy - d, cx - d
            else:
                # We've reached the start again
                return code

        visited.add((cy, cx))
        iter_count += 1

    return code


def compute_codes(images: pd.DataFrame, name: str) -> None:
    """
    Computes and stores the freeman codes for the
    :param images: The images to compute the freeman codes for
    :param name: The name of resulting file the codes are saved to
    """
    code_list = []
    # Calculate Freeman codes for each image
    for index, row in images.iterrows():
        label = row[1]
        pixel_data = row[2:]
        code = get_freeman_code(pixel_data.to_numpy(dtype=np.uint8).reshape((BIG_IMG_SIZE, BIG_IMG_SIZE)))
        code_list.append({'digit': label, 'code': code})

    # Store the Freeman codes as a CSV
    freeman_codes = pd.DataFrame(code_list, columns=['digit', 'code'])
    freeman_codes.to_csv(name, index=False)


def compute_distances(codes: pd.DataFrame, selected_code: str) -> List[dict]:
    """
    Compute and return an array of Levenshtein distances given a freeman code
    :param codes: The set of training freeman codes
    :param selected_code: The selected freeman code to compute distances to
    :return: An array of Levenshtein distances
    """
    distances: List[dict] = []
    # Calculate Levenshtein distances for every stored digit
    for index, row in codes.iterrows():
        dist = Levenshtein.distance(row[1], selected_code)
        distances.append({
            'digit': row[0],
            'distance': dist
        })
    return distances


def custom_KNN(codes: pd.DataFrame, selected_code: str) -> int:
    """
    Custom implmentation of the K-Nearest-Neighbors algorithm
    :param codes: DataFrame containing labels and Freeman codes
    :param selected_code: The Freeman code to compute the distances to
    :return: The most common label of the K nearest neighbors (the prediction result)
    """
    distances = compute_distances(codes, selected_code)
    sorted_dists = sorted(distances, key=lambda item: item['distance'])
    counts = Counter(map(lambda x: x['digit'], sorted_dists[:10]))
    guess = max(counts, key=counts.get)

    return guess


def log(msg: str):
    print(f'[{datetime.now().time()}]: {msg}')


if __name__ == '__main__':
    log('Program started')
    args = parser.parse_args()

    if args.p:
        # Divide data into training and test sets
        total_data = pd.read_csv('data/train.csv')
        training_data = total_data.sample(n=2000, random_state=1)
        test_data = total_data.drop(training_data.index).sample(n=1000, random_state=2)
        training_data.to_csv('data/training_set.csv')
        test_data.to_csv('data/test_set.csv')

        # Run image pre-processing
        preprocess_data(training_data, 'data/processed_train.csv')
        preprocess_data(test_data, 'data/processed_test.csv')
        log('Finished preprocessing data')
        sys.exit(0)

    # Read processed image data to use for display
    processed_test = pd.read_csv('data/processed_test.csv')
    log('Loaded preprocessed data')

    if args.c:
        # Compute and store Freeman codes
        processed_train = pd.read_csv('data/processed_train.csv')

        compute_codes(processed_train, 'data/freeman_codes_train.csv')
        compute_codes(processed_test, 'data/freeman_codes_test.csv')
        log('Finished computing Freeman codes')
        sys.exit(0)

    # Compute Levenshtein distance for stored Freeman Codes
    train_codes = pd.read_csv('data/freeman_codes_train.csv')

    if args.s:
        """Demo entire process on a single image file"""
        # Read in image file
        single_img = cv.imread(args.s, cv.IMREAD_GRAYSCALE)
        # Run preprocessing steps
        processed = process_image(single_img)
        # Compute Freeman code
        f_code = get_freeman_code(processed)
        # Run KNN classifier
        pred = custom_KNN(train_codes, f_code)
        # Print prediction
        print(f'Prediction: {pred}')
        sys.exit(0)

    test_codes = pd.read_csv('data/freeman_codes_test.csv')

    log('Starting test set predictions')
    # Do predictions for test set
    correct: dict[int, list] = {}
    wrong: dict[int, list] = {}
    for index, row in test_codes.iterrows():
        sel_digit, sel_string = row
        guess = custom_KNN(train_codes, sel_string)

        if guess != sel_digit:
            if sel_digit not in wrong:
                wrong[sel_digit] = []
            wrong[sel_digit].append((index, guess))
        else:
            if sel_digit not in correct:
                correct[sel_digit] = []
            correct[sel_digit].append(index)

    print('Results:')
    total_wrong = 0
    for i in range(10):
        num_c = len(correct[i])
        num_w = len(wrong[i])
        total_wrong += num_w
        print(f'[Accuracy {i}]: {num_c}/{num_w + num_c} ({num_c/(num_w + num_c)})')

    print(f'Total accuracy: {1000-total_wrong}/1000 ({(1000-total_wrong)/1000})')
    print(wrong)

    log('Finished evaluating test data')

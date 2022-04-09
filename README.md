# Digit-Recognition README
By: Frank Yu 101076416

Github repo: https://github.com/doodlehead/Digit-Recognition

## Environment setup
- Run with Python 3.9 or more recent
  - Make sure your version of Python comes with pip, it will be used to install the dependencies
- Navigate to the top level of the project folder where the `requirements.txt` file is found and install the python dependencies using the command:
  - `pip install -r requirements.txt`

## How to run
- The complete dataset is NOT included in the submission as specified in the instructions
- If you wish to test the entire process with the original dataset, you can download the data individually here:
  - https://github.com/doodlehead/Digit-Recognition/blob/master/data/train.csv (Place `train.csv` in `data/`)
- Alternatively you can clone/download the entire github repo which contains all the required data.
- With only the data provided in the submission you can run the script with the `-s` flag and with no flags.
  - For the `-p` and `-c` use cases you'll need to download the entire dataset.
- Run the main script using the command:
  - `python .\DigitRecognition.py {-p | -c | -s <filename>}`

### Flags:
- Note: the flags are all mutually exclusive
- `-p`: Run the preprocessing step.
  - **Make sure the `train.csv` file is present in the `data` folder**
- `-c`: Run the Freeman code calculation step. Make sure you've previously run the preprocessing steps.
  - The files `processed_test.csv` and `processed_train.csv` should be both be present in the `data` folder.
- No flags: Run the recognition system testing on the entire test set of 1000 digits.
  - **NOTE: This runs for over 3 minutes**
- `-s`: Run the entire process on a single test image
  - Samples to use with this command are included in the `data/samples/` folder

### Examples:
- `python .\DigitRecognition.py -s data/samples/345.png`
  - Run the digit recognition system on a single test image with the path `data/samples/345.png`
- `python .\DigitRecognition.py -p`
  - Run the preprocessing step. Generates files in the `data/` folder
- `python .\DigitRecognition.py -c`
  - Run the Freeman code calculation step. Generates files in the `data/` folder.
- `python .\DigitRecognition.py`
  - Run the recognition system testing on the entire test set of 1000 digits (**NOTE: This runs for over 3 minutes**)

## Expected results
- Running with the `-s` flag:
  - The program will print a prediction of what digit it thinks the provided image is.
- Running with no flags:
  - The program will run for over 3 minutes and then print out the accuracy statistics
- Running with the `-p` flag:
  - Generates the files in `data/`:
    - training_set.csv
    - test_set.csv
    - processed_train.csv
    - processed_test.csv
- Running with the `-c` flag:
  - Generates the files in `data/`
    - freeman_codes_train.csv
    - freeman_codes_test.csv

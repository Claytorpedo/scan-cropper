# scan-cropper
Find photos in scanned images, then orient and crop them.

A quick python script designed to find photographs in scanned images (in particular to find rectangular images on a white background).
It is designed to find multiple photos at a time.
It both crops and orients them, so scanned photos don't have to be perfectly straight.

## Example
You could scan a photo album 3 photos at a time, leaving some space between them on the scanner.
Run the script with `--photos-per-scan 3` on your scanned files.
For each file it processes, it should output 3 cropped and oriented photos from the scan.

## Requirements
* Python 3
* numpy         `pip install numpy`
* python-opencv `pip install python-opencv`

## Usage
`python scan_cropper.py` will run it on pictures in the current directory, putting results in an `output` directory.

Run `python scan_cropper.py -h` to see a list of input options.

`RUN_CROPPING_PROGRAM.bat` is included to make it easier to use for non tech-savvy users.
Can be double-clicked like a regular program, and will run the script with sensible default options.

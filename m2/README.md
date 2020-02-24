# File & Image Converter

## Introduction

Different from **file & QR-code converter**, the image carries much more information than the QR-code.

| Compare           | QR-Code              | Image         |
| ----------------- | -------------------- | ------------- |
| Size [Byte/Image] | no more than `< 6kB` | up to `> 1MB` |
| Carrier           | camera, screen-shot  | screen-shot   |

## Requirement

The python library of `pillow` and `pypng`.

## Usage

### Argument

* `--input`: the input file of converting to image, or folders of images to decode.
* `--output`: the output folder for image, or target file to export from image.
* `--cpu_number`: the number of CPU to process files.
* `--level`: the quality level of image: 1, 2, 3, 4; larger level number will carry more data.
* `--image_size`: image size in pixel.

## Demo

Same as File & QR-Code Converter.

...

## Disclaimer

...
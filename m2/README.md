# File & Image Converter

## Introduction

Different from **file & QR-code converter**, the image carries much more information than the QR-code.

| Compare           | QR-Code            | Image        |
| ----------------- | ------------------ | ------------ |
| Size [Byte/Image] | `< 6kB`            | up to `1MB+` |
| Carrier           | camera, screenshot | screenshot   |

## Requirement

The python library of `pillow` and `pypng`.

## Usage

### Argument

* `--input`: the input file of converting to image, or folders of images to decode.
* `--output`: the output folder for image, or target file to export from image.
* `--cpu_number`: the number of CPU to process files.
* `--level`: the quality level of image: 1, 2, 3, 4; larger level number will carry more data.
* `--compress`: the encoding compression level: 0 ~ 9 or -1 as default.
* `--image_size`: image size in pixel; please follow 2 numbers; if given one image size, the image would be square; if one or all parameter are `<=0`, it will convert into one image but to consider this image size are not given.

## Warning

Do not use **black background** to take the screenshot!

## Demo

Same as **File & QR-Code Converter**.

## Disclaimer

...
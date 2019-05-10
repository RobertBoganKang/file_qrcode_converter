# File & QR-Code Converter
## Introduction
For some reason, some file (such as document, script, or pictures) cannot be copied from any computer (for example, from companies, organizations or government). This project allows us to export small file through QR Code:)

Unfortunately, it can support only one file of conversion (be sure to pack all files in only one file). 

Only small files are allowed since one QR Code can only store around or less than 1kB data (maximum of 65536 qr-code are supported).

It is safe to use since multiple way of file checking will ensure NO DATA WILL LOSS from any conversion.
### Taking pictures
Use a 1 to 1 ratio of pictures when taking pictures from screen can achieve better result.
## Requirement
The python library of `PIL`, `pyqrcode` and `pyzbar` are required.
## Usage
### common
* `--input`: the input file of converting to qr-code, or folders of images to decode.
* `--output`: the output folder for qr-code, or target file to export from qr-code.
* `--chunk_size`: the number of byte for each data to store for each qr-code.
* `--cpu_number`: the number of cpu to process files.
### qr.py
* `--quality`: error allowance, L, M, Q, H from low to high.
### jab.py
* `--image_size`: the size of image `(1~32)`.
* `--color_number`: the color number of image `2^[2~8]`.
* `--quality`: error allowance `(1~10)`.
## Demo
Type `python xxx.py -i demo -o mozart_11.mid`, where `xxx` represents `qr` or `jab`, it converts one or multiple qr-code images from the folder `demo` to the `mid` music file -- `mozart_11.mid` is the *Mozart Piano Sonata No.11 (first part)* I played.

Of course, the script `python xxx.py -o demo -i mozart_11.mid` is the reverse way of converting back from the file to multiple qr-code images.
## Disclaimer
This project can only be used under the law of the state or country from any user.

However, it can be used for reasearch purposes without penalty.

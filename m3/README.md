# File & Log Text Converter

## Introduction

If log text saving is available, log files can carry a lot of information for transferring files.

## Usage

### Argument

* `--input`: the source file path to encode, or log text file path to decode.
* `--output`: the target folder to decode.
* `--row_size`: the size of bytes to encode for each row.
* `--command`: (optional) the command string pattern.
* `--compress_level`: (optional) the compression level from -1 to 9.
* `--encoding`: (optional) the encoding of log file to decode, if not given -> auto detect.

## Usage

### Encode

```bash
python convert.py -i <file_to_encode>
```

The terminal logs should be recorded at the same time.

### Decode

```bash
python convert.py -i <log_text_file_to_decode> -o <folder_to_export_decoded_file>
```

Demo with `python convert.py -i demo/example.log -o result` in this project.

### Cheat Sheet

- The log text file should be fully recorded.
- When multiple files are encoded, files of different files should have different filenames.
- Do not press any keys while pushing the log.
- The larger the text within the terminal, the faster the logging.

## Disclaimer

...
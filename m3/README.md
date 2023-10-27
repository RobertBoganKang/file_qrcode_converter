# File & Log Text Converter

## Introduction

If the log text saving is availabel, the log file could carry much information.

## Usage

### Argument

* `--input`: the source file path to encode, or log text file path to decode.
* `--output`: the target folder to decode.
* `--line_size_limit`: the size of bytes to encode for each line.
* `--command_string`: the string to check as start/end command.

## Usage

### Encode

```bash
python3 convert.py -i <file_to_encode>
```

Meanwhile, the terminal log should be recorded at the same time.

### Decode

```bash
python3 convert.py -i <log_text_file_to_decode> -o <folder_to_export_decoded_file>
```

### Warning

- The log text file should be fully recorded.
- The file for different file should have different file name, when multiple file encoded.

## Disclaimer

...
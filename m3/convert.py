import argparse
import base64
import os
import zlib


class Log2File(object):
    """
    decode log text to file
    """

    def __init__(self, ops):
        self.input = ops.input
        self.output = ops.output
        self.encoding = ops.encoding
        self.rbk = ops.command
        assert len(self.rbk) > 0

    @staticmethod
    def add_numbered_suffix(path):
        if not os.path.exists(path):
            return path

        base, ext = os.path.splitext(path)
        counter = 1
        new_path = f"{base}({counter}){ext}"
        while os.path.exists(new_path):
            counter += 1
            new_path = f"{base}({counter}){ext}"
        return new_path

    def remove_file(self, path, check=True):
        if os.path.exists(path):
            if check:
                while True:
                    ck = input(f'INFO: do you wish to replace `{path}`, [yes/no]: ').lower()
                    if ck == 'yes':
                        os.remove(path)
                        return path
                    elif ck == 'no':
                        return self.add_numbered_suffix(path)
                    else:
                        continue
            else:
                os.remove(path)
                return path
        else:
            return path

    @staticmethod
    def string_to_bytes(s):
        original_bytes = zlib.decompress(base64.b85decode(s.encode()))
        return original_bytes[0], original_bytes[1:]

    @staticmethod
    def string_to_path(string):
        # noinspection PyBroadException
        try:
            decoded_path = zlib.decompress(base64.b85decode(string.encode())).decode()
            return decoded_path
        except Exception:
            return None

    def write_byte(self, string, row_num, w):
        if len(string) > 0:
            num, byte = self.string_to_bytes(string)
            # row number check
            if (row_num + 1) % 256 == num:
                w.write(byte)
                return num
            else:
                raise ValueError('ERROR: row number is not continuous!')
        else:
            raise ValueError('ERROR: string should not empty!')

    @staticmethod
    def check_command(line, rbk='rbk'):
        """
        to check the command lines:
            * `rbkRBKr...` is starting command, return 1;
            * `RBKrbkR...` is ending command, return -1;
            * else not a command, return 0;
        :param line: each line maybe a command
        :param rbk: a sample word of command indicator;
        :return: [-1, 0, 1]
        """
        if len(line) == 0:
            return 0
        rbk_length = len(rbk)
        for i in range(len(line)):
            c = line[i].lower()
            idx = i % rbk_length
            if c != '\n' and c != rbk[idx]:
                return 0
        if line[0] == rbk[0].upper():
            return -1
        if line[0] == rbk[0].lower():
            return 1

    def convert(self):
        out_folder = self.output
        if self.encoding is None or self.encoding == '':
            # noinspection PyBroadException
            try:
                import chardet
                # guess the encoding of log from the first byte
                with open(self.input, 'rb') as f:
                    encoding_type = chardet.detect(f.read())['encoding']
                    if encoding_type is not None:
                        print(f'INFO: guess file encoding is `{encoding_type}`')
            except Exception:
                encoding_type = input('ERROR: package `chardet` not found!\n'
                                      'Please enter the encoding type: ').strip()
        else:
            encoding_type = self.encoding
        with open(self.input, 'r', errors='ignore', encoding=encoding_type) as f:
            line = f.readline()
            while line:
                string = line.strip()
                outer_string_status = self.check_command(string, rbk=self.rbk)
                # find the starting point
                if outer_string_status != 1:
                    # if not found, go to next line
                    line = f.readline()
                else:
                    # if found, then process
                    # read header: 1st line is the file name
                    line = f.readline()
                    string = line.strip()
                    os.makedirs(out_folder, exist_ok=True)
                    file_name = self.string_to_path(string)
                    # if file name cannot be decoded
                    if file_name is None:
                        continue
                    # if file name can be decoded, then process
                    path = os.path.join(out_folder, file_name)
                    print(f'INFO: now decode `{path}`')
                    path = self.remove_file(path, check=True)
                    row_num = -1
                    with open(path, 'ab') as w:
                        while line:
                            line = f.readline()
                            string = line.strip()
                            string_status = self.check_command(string, rbk=self.rbk)
                            if string_status == 0:
                                # noinspection PyBroadException
                                try:
                                    row_num = self.write_byte(string, row_num, w)
                                except Exception:
                                    print(f'ERROR: `{path}` error, remove file and search next!')
                                    self.remove_file(path)
                                    break
                            elif string_status == -1:
                                # break inner while loop
                                print(f'INFO: `{path}` decode successfully.')
                                break
                            elif string_status == 1:
                                print(f'ERROR: ending command not found, remove `{path}`!')
                                self.remove_file(path)
                                break


class File2Log(object):
    """
    encode file to log text
    """

    def __init__(self, ops):
        self.input = ops.input
        self.limit = ops.row_size
        self.rbk = ops.command
        self.zlib_level = ops.compress_level

        assert len(self.rbk) > 0
        self.check_zlib_level()

    def check_zlib_level(self):
        assert -1 <= self.zlib_level <= 9

    @staticmethod
    def build_command(command, length, rbk='rbk'):
        """
        to build the command lines:
            * `rbkRBKr...` is starting command, return 1;
            * `RBKrbkR...` is ending command, return -1;
        :param command: 1 or -1
        :param length: command length
        :param rbk: a sample word of command indicator;
        :return: `rbkRBKr...` or `RBKrbkR...`
        """
        line = ''
        rbk_length = len(rbk)
        if command == 1:
            status = True
        elif command == -1:
            status = False
        else:
            raise ValueError('ERROR: command should be either -1 or 1!')
        for i in range(length):
            idx = i % rbk_length
            if i % rbk_length == 0:
                status ^= True
            if status:
                line += rbk[idx].upper()
            else:
                line += rbk[idx].lower()
        return line

    def bytes_to_string(self, b, row_num):
        """
        first byte to be the row number index of check
        """
        return base64.b85encode(zlib.compress(bytes([row_num % 256]) + b), self.zlib_level).decode()

    @staticmethod
    def path_to_string(path):
        name = os.path.split(path)[-1]
        return base64.b85encode(zlib.compress(name.encode(), 9)).decode()

    def convert(self):
        path = self.input
        try:
            # print start command
            print(self.build_command(1, self.limit, rbk=self.rbk))
            # print file name
            print(self.path_to_string(path))
            # encode contents
            row_bytes = b''
            row_num = 0
            with open(path, 'rb') as f:
                byte = f.read(1)
                row_bytes += byte
                i = 1
                while byte:
                    i += 1
                    # Do stuff with byte.
                    byte = f.read(1)
                    row_bytes += byte
                    if i % self.limit == 0:
                        i = 0
                        string = self.bytes_to_string(row_bytes, row_num)
                        row_num += 1
                        print(string)
                        row_bytes = b''
            if len(row_bytes) != 0:
                string = self.bytes_to_string(row_bytes, row_num)
                print(string)
            # print end command
            print(self.build_command(-1, self.limit, rbk=self.rbk))
        except IOError:
            IOError('ERROR: error while opening the file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert log text to file')
    # argument for project
    io_group = parser.add_argument_group('file i/o')
    io_group.add_argument('--input', '-i', type=str,
                          help='the source file path to encode, log text file path to decode',
                          required=True)
    io_group.add_argument('--output', '-o', type=str, help='the target folder to decode', default=None)

    # argument for log
    log_group = parser.add_argument_group('log arguments')
    log_group.add_argument('--row_size', '-s', type=int, help='the size of bytes to encode for each row',
                           default=256)
    log_group.add_argument('--command', '-c', type=str, help='(optional) the command string pattern', default='rbk')
    log_group.add_argument('--compress_level', '-z', type=int,
                           help='(optional) the compression level from -1 to 9', default=-1)
    log_group.add_argument('--encoding', '-e', type=str,
                           help='(optional) the encoding of log file to decode, if not given -> auto detect',
                           default=None)

    args = parser.parse_args()

    if args.output is None:
        f2l = File2Log(args)
        f2l.convert()
    else:
        l2f = Log2File(args)
        l2f.convert()

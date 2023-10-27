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
        self.rbk = 'rbk'

    @staticmethod
    def remove_file(path):
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def string_to_bytes(s):
        return zlib.decompress(base64.b85decode(s.encode()))

    @staticmethod
    def string_to_path(string):
        return zlib.decompress(base64.b85decode(string.encode())).decode()

    def write_byte(self, string, w):
        if len(string) > 0:
            w.write(self.string_to_bytes(string))

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
        with open(self.input, 'r') as f:
            line = f.readline()
            while line:
                string = line.strip()
                outer_string_status = self.check_command(string, rbk=self.rbk)
                if outer_string_status == 1:
                    # read header: file name
                    line = f.readline()
                    string = line.strip()
                    os.makedirs(out_folder, exist_ok=True)
                    path = os.path.join(out_folder, self.string_to_path(string))
                    print(f'INFO: now decode `{path}`')
                    self.remove_file(path)
                    with open(path, 'ab') as w:
                        while line:
                            line = f.readline()
                            string = line.strip()
                            string_status = self.check_command(string, rbk=self.rbk)
                            if string_status == 0:
                                # noinspection PyBroadException
                                try:
                                    self.write_byte(string, w)
                                except Exception:
                                    print(f'ERROR: `{path}` error, search next!')
                                    break
                            elif string_status == -1:
                                # break inner while loop
                                print(f'INFO: `{path}` decode successfully.')
                                break
                            elif string_status == 1:
                                print(f'ERROR: ending command not found, remove `{path}`!')
                                self.remove_file(path)
                                break
                else:
                    # go to next line
                    line = f.readline()


class File2Log(object):
    """
    encode file to log text
    """

    def __init__(self, ops):
        self.input = ops.input
        self.limit = ops.line_size_limit
        self.rbk = 'rbk'

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

    @staticmethod
    def bytes_to_string(b):
        return base64.b85encode(zlib.compress(b)).decode()

    @staticmethod
    def path_to_string(path):
        name = os.path.split(path)[-1]
        return base64.b85encode(zlib.compress(name.encode())).decode()

    def convert(self):
        path = self.input
        try:
            # print start command
            print(self.build_command(1, self.limit, rbk=self.rbk))
            # print file name
            print(self.path_to_string(path))
            row_bytes = b''
            with open(path, 'rb+') as f:
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
                        string = self.bytes_to_string(row_bytes)
                        print(string)
                        row_bytes = b''
            if len(row_bytes) != 0:
                string = self.bytes_to_string(row_bytes)
                print(string)
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

    # argument for image
    log_group = parser.add_argument_group('log arguments')
    log_group.add_argument('--line_size_limit', '-s', type=int, help='the size of bytes to encode for each line',
                           default=256)

    args = parser.parse_args()

    if args.output is None:
        f2l = File2Log(args)
        f2l.convert()
    else:
        l2f = Log2File(args)
        l2f.convert()

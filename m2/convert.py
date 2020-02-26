import argparse
import multiprocessing
import multiprocessing as mp
import os
import shutil
import zlib

import numpy as np
from PIL import Image


class Common(object):
    """
    Description of `level`:
        Level 1: `1-bit` * `8-digits` == `1-Byte`
        Level 2: `2-bit` * `4-digits` == `1-Byte`
        Level 3: `4-bit` * `2-digits` == `1-Byte`
        Level 4: `8-bit` * `1-digit` == `1-Byte`
    ----------------------------------------------
    Description of `image_size`:
        the decoded image size is `image_size`
    ----------------------------------------------
    Description of `header` in image pixels,
        given image as an example:
        * 6 pixels store the information of `index`
            and `level` for encoding and decoding process.
            * `16_bits` stores the index [0, 65535].
            * `2_bits` stores the level [0, 3].
    ----------------------------------------------
    Description of header info of data (store file name):
        [256 Bytes store 256 characters] to store file name
    """

    def __init__(self):
        self.level = None
        self.image_size = None

        # initialize
        self.n_digits = None
        self.n_base = None
        self.n_gap = None
        self.image_data_carry = None
        self.last_empty_data_carry = None

    def initialize_level(self, level=2):
        """ initialize level parameters """
        self.level = level
        # calculate other parameters
        self.n_digits = int(8 / 2 ** (self.level - 1))
        self.n_base = 2 ** (2 ** (self.level - 1))
        self.n_gap = 255 / (self.n_base - 1)

    def initialize_image_size(self, image_size=(1200, 800)):
        """ initialize image_size parameters """
        self.image_size = image_size
        # given six `1_bit` * `3-channels` pixels as header to store info
        # 3-channels * 6-pixels = 16_bit-index + 2_bit-level == 18_bits
        self.image_data_carry = (self.image_size[0] * self.image_size[1] - 6) * 3
        # crop it to store Bytes not bit splits, the rest will fill with empty white pixels
        # the number of last data to be empty
        self.last_empty_data_carry = self.image_data_carry - self.image_data_carry // self.n_digits * self.n_digits
        # the number of data for each image to carry
        self.image_data_carry -= self.last_empty_data_carry

    @staticmethod
    def cpu_count(cpu):
        """
        get the cpu number
        :return: int; valid cpu number
        """
        max_cpu = multiprocessing.cpu_count()
        if 0 < cpu <= max_cpu:
            return cpu
        elif cpu == 0 or cpu > max_cpu:
            return max_cpu
        elif 1 - max_cpu < cpu < 0:
            return max_cpu + cpu
        else:
            return 1

    @staticmethod
    def fix_out_path(input_path, output_path, encode=True):
        """
        the default folder is set to `#__[<input_name>]__#`.
        """
        input_path = os.path.abspath(input_path)
        if output_path is None:
            name = os.path.split(input_path)[1]
            if encode:
                out_path = os.path.join(os.path.dirname(input_path), '#__(' + name + ')__#')
            else:
                out_path = os.path.join(os.path.dirname(input_path), '#__[' + name + ']__#')
            os.makedirs(out_path, exist_ok=True)
        else:
            out_path = os.path.abspath(output_path)
        return out_path


class File2Image(Common):
    """ encoder """

    def __init__(self, ops):
        super().__init__()
        self.input = ops.input
        self.output = self.fix_out_path(self.input, ops.output)
        self.cpu_number = self.cpu_count(ops.cpu_number)

        # initialize with given parameters
        self.initialize_level(level=ops.level)
        self.initialize_image_size(image_size=ops.image_size)

        # fix input path
        self.check_output_folder()

    def ask_for_new_path(self):
        new_path = input('please type new path for exporting data-image:')
        self.output = new_path

    def check_output_folder(self):
        # if path not exist, make directory and return
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
            return

        if not os.path.isdir(self.output):
            self.ask_for_new_path()
            self.check_output_folder()
        else:
            names = os.listdir(self.output)
            if len(names) != 0:
                while True:
                    check = input('the folder is not empty, do you wish to remove them [y/n]: ')
                    if check == '':
                        continue
                    if check.strip().lower() in ['yes', 'y']:
                        shutil.rmtree(self.output)
                        os.makedirs(self.output)
                        return
                    elif check.strip().lower() in ['no', 'n']:
                        self.ask_for_new_path()
                        self.check_output_folder()
                        return
                    else:
                        print('input not recognized!')

    @staticmethod
    def to_base(integer, base, length):
        """ convert 8-bit integer into list of splits """
        split_list = []
        i = 0
        while i < length:
            split_list.append(integer % base)
            integer = integer // base
            i += 1
        split_list.reverse()
        return split_list

    @staticmethod
    def path_to_bytes(path):
        path = os.path.split(path)[1]
        b = path.encode('utf-8')
        for _ in range(256 - len(b)):
            b += b'\x00'
        return b

    def split_array(self, array):
        """
        split the `1-Byte` or `8-bit` into several chunks of data
        @param array: list(int); range of integer is `0 ~ 255`
        """
        array_list = []

        for a in array:
            array_list += self.to_base(a, self.n_base, self.n_digits)
        return array_list

    def encode_header(self, i):
        result = self.to_base(i, 2, 16) + self.to_base(self.level - 1, 2, 2)
        return [255 - x * 255 for x in result]

    def data_to_pixel(self, num):
        return round(self.n_gap * num)

    @staticmethod
    def add_frame(matrix, pixel):
        matrix_rebuild = np.array(
            [[pixel for _ in range(len(matrix[0]) + 2)] for _ in range(len(matrix) + 2)]
        )
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix_rebuild[i + 1, j + 1] = matrix[i, j]
        return np.array(matrix_rebuild).astype(np.uint8)

    def array_transform(self, image_array):
        image_reshaped = np.reshape(image_array,
                                    newshape=(self.image_size[1], self.image_size[0], 3))
        # add black frame
        return self.add_frame(image_reshaped, [0, 0, 0])

    def export_image_helper(self, image_array, i, out_folder):
        image_reshaped = self.array_transform(image_array)
        img = Image.fromarray(image_reshaped)
        out_image_path = os.path.join(out_folder, str(i) + '.png')
        img.save(out_image_path)
        print(f'[{out_image_path}] has been exported ~')

    def encode_image_single(self, bundles):
        array_list, out_folder, i = bundles
        image_array = np.array(self.encode_header(i)
                               + [self.data_to_pixel(array_list[ii]) for ii in
                                  range(i * self.image_data_carry, (i + 1) * self.image_data_carry)]
                               + [255 for _ in range(self.last_empty_data_carry)])
        self.export_image_helper(image_array, i, out_folder)

    @staticmethod
    def rbk_background(s1, s2):
        """
        generate `rbk` background
        @param s1: int; image size 0
        @param s2: int; image size 1
        @return: numpy array of image
        """
        rbk = 'rrr..bbb..k..k.....b...k...,' \
              'r..r.b..b.k.k..r.r.bb..k.k.,' \
              'rrr..bbb..kk...rr..b.b.kk..,' \
              'r.r..b..b.k.k..r...b.b.k.k.,' \
              'r..r.bbb..k..k.r...bb..k.k.,' \
              '...........................'

        rbk = rbk.split(',')
        r_color = [255, 0, 0]
        b_color = [0, 0, 255]
        k_color = [64, 64, 64]
        dot_color = [255, 255, 255]
        rbk_size = [len(rbk), len(rbk[0])]
        table = []
        for i in range(s2):
            row = []
            for j in range(s1):
                i_idx = i % rbk_size[0]
                row_number = i // rbk_size[0]
                j_idx = (j - 10 * row_number) % rbk_size[1]
                if rbk[i_idx][j_idx] == 'r':
                    row.append(r_color)
                elif rbk[i_idx][j_idx] == 'b':
                    row.append(b_color)
                elif rbk[i_idx][j_idx] == 'k':
                    row.append(k_color)
                else:
                    row.append(dot_color)
            table.append(row)
        table = np.array(table).flatten()
        return table

    def encode_image(self, array_list, out_folder):
        """
        split image
        @param array_list: list(int); data to encode
        @param out_folder: str; exported output folder
        @return: None
        """
        # full sized image
        i_length = len(array_list) // self.image_data_carry
        # if file too big for index
        if i_length > 65535:
            raise OverflowError('file is too big ~')
        bundles = [[array_list, out_folder, x] for x in range(i_length)]
        # encode images
        with mp.Pool(self.cpu_number) as pool:
            pool.map(self.encode_image_single, bundles)
        last_array_list = array_list[i_length * self.image_data_carry:]
        # encode image for rest of data
        if len(last_array_list) != 0:
            rbk_array = self.rbk_background(self.image_size[0], self.image_size[1])
            image_array = np.array(
                self.encode_header(i_length)
                + [self.data_to_pixel(ii) for ii in last_array_list]
                # final empty data was all white pixels
                # compress algorithm will not consider data at the end
                # add rbk background
                + [rbk_array[i] for i in
                   range(self.image_size[0] * self.image_size[1] * 3
                         - (self.image_data_carry - len(last_array_list) + self.last_empty_data_carry),
                         self.image_size[0] * self.image_size[1] * 3
                         )
                   ]
            )
            self.export_image_helper(image_array, i_length, out_folder)

    def encode(self):
        with open(self.input, 'rb') as f:
            b_arr = f.read()
            # add file name
            name_bytes = self.path_to_bytes(self.input)
            # compress
            b_compress = zlib.compress(name_bytes + bytes(b_arr), 9)
            compressed_byte_list = list(b_compress)
            encoded_data = self.split_array(compressed_byte_list)
            os.makedirs(self.output, exist_ok=True)
            self.encode_image(encoded_data, self.output)


class SingleImage2TempFile(Common):
    """ decode one image to temp file """

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_base(integer_array, base):
        """ convert list of splits into 8-bit integer"""
        integer = 0
        for i in integer_array:
            integer *= base
            integer += i
        return int(integer)

    def combine_array(self, array):
        combined_list = []
        for i in range(len(array) // self.n_digits):
            combined_list.append(
                self.from_base(array[i * self.n_digits:(i + 1) * self.n_digits], self.n_base)
            )
        return combined_list

    def pixel_to_data(self, num):
        return round(num / self.n_gap)

    def decode_header(self, array):
        array = [round(1 - x / 255) for x in array]
        return self.from_base(array[:16], 2), self.from_base(array[16:18], 2) + 1

    @staticmethod
    def zig_zag_traversal_find_upper_left_corner(matrix, identifier, tolerance):
        """
        [https://www.geeksforgeeks.org/print-matrix-zag-zag-fashion/]
        zig-zag traversal to find identifier for the first time
        -------------------------------------------------------------
        o///.............
        //X---------+....
        /.|ooooooooo|....
        ..+---------+....
        .................
        -------------------------------------------------------------
        namely: from `o`, use zig-zag traversal
            to find the upper-left corner pixel (find target `X`)
        @param matrix: numpy 2D array of integer
        @param identifier: list(list(int))
        @param tolerance: the tolerance of pixels to consider as target
        @return: None or list(int); none or index
        """
        identifier = np.array(identifier)
        rows = len(matrix)
        columns = len(matrix[0])
        solution = [[] for _ in range(rows + columns - 1)]
        for i in range(rows):
            for j in range(columns):
                total = i + j
                # find result, check corner
                # noinspection PyChainedComparisons
                if (
                        np.mean(np.abs(matrix[i, j] - identifier)) < tolerance
                        and np.mean(np.abs(matrix[i + 1, j] - identifier)) < tolerance
                        and np.mean(np.abs(matrix[i, j + 1] - identifier)) < tolerance
                        # and np.mean(np.abs(matrix[i + 1, j + 1] - identifier)) > tolerance
                ):
                    return i, j
                if total % 2 == 0:
                    # add at beginning
                    solution[total].insert(0, matrix[i, j])
                else:
                    # add at end of the list
                    solution[total].append(matrix[i, j])
        return None

    @staticmethod
    def find_bottom_right_corner_crop_image(matrix, target, idx, tolerance):
        """
        find the bottom-right pixel index from upper-left pixel index \
            and crop the image.
        -------------------------------------------------------------
        .................
        ..o>>>>>>>>>*....
        ..|oooooooooV....
        ..+---------X....
        .................
        -------------------------------------------------------------
        namely: from upper-left corner pixel \
            goes right from `>` edge first \
            and goes down from `V` edge to find target `X`
        @param matrix: image matrix
        @param target: the target pixel color to follow as frame
        @param idx: upper-left corner coordinate index
        @param tolerance: the tolerance of pixels to consider as target
        @return: matrix of cropped image
        """
        i, j = idx
        # find to the right
        while i < len(matrix) - 1:
            if np.mean(np.abs(matrix[i + 1, j] - target)) < tolerance:
                i += 1
            else:
                # check corner
                if np.mean(np.abs(matrix[i, j + 1] - target)) < tolerance:
                    break
        while j < len(matrix[0]) - 1:
            if np.mean(np.abs(matrix[i, j + 1] - target)) < tolerance:
                j += 1
            else:
                # check corner
                if np.mean(np.abs(matrix[i - 1, j] - target)) < tolerance:
                    break
        return matrix[idx[0] + 1:i, idx[1] + 1:j]

    def image_path_to_data(self, path):
        """
        read path and extract the effective data
        @param path: str
        @return: None
        """
        # read image
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        # set 16 tolerance to find corner of black frame
        # find upper-left corner
        index = self.zig_zag_traversal_find_upper_left_corner(img, [0, 0, 0], 16)
        if index is not None:
            # set 32 as tolerance to find edges
            cropped_image = self.find_bottom_right_corner_crop_image(img, [0, 0, 0], index, 32)
            return cropped_image.flatten()
        else:
            raise ValueError(f'[{path}] cannot be decoded!')

    def path_decode_to_temp_data(self, path):
        # extract data from image
        data = self.image_path_to_data(path)
        index, level = self.decode_header(data[:18])
        # get all parameters for decoding
        self.initialize_level(level)
        decoded_data = self.combine_array([self.pixel_to_data(x) for x in data[18:]])
        return index, decoded_data

    def decode_to_temp_file(self, path):
        index, decoded_data = self.path_decode_to_temp_data(path)
        # export data file
        out_path = os.path.join(os.path.dirname(path), str(index) + '.tmp')
        with open(out_path, 'wb') as w:
            w.write(bytes(decoded_data))
        print(f'[{path}] has been decoded ~')


class Image2File(object):
    """ decoder """

    def __init__(self, ops):
        self.input = ops.input
        self.output = self.output = Common.fix_out_path(self.input, ops.output, encode=False)
        self.cpu_number = Common.cpu_count(ops.cpu_number)

    @staticmethod
    def bytes_to_file_name(b):
        d = [x for x in b if x != 0]
        return bytes(d).decode('utf-8')

    @staticmethod
    def decode_image_to_temp_file_single(path):
        # create new object for each image of decoding
        decode = SingleImage2TempFile()
        decode.decode_to_temp_file(path)

    def decode_image_to_temp_file(self):
        image_files = os.listdir(self.input)
        image_files = [os.path.join(self.input, x) for x in image_files if not x.endswith('.tmp')]
        with mp.Pool(self.cpu_number) as pool:
            pool.map(self.decode_image_to_temp_file_single, image_files)

    def decode(self):
        # decode data
        self.decode_image_to_temp_file()
        # combine temp data files
        byte_files = os.listdir(self.input)
        byte_files = [os.path.join(self.input, x) for x in byte_files if x.endswith('.tmp')]
        # find ending number
        numbers = [int(os.path.splitext(os.path.split(x)[1])[0]) for x in byte_files]
        numbers.sort()
        data_combine = bytearray()
        for i in range(numbers[-1] + 1):
            data_file_path = os.path.join(self.input, str(i) + '.tmp')
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f'[{data_file_path}] data file not found!')
            with open(data_file_path, 'rb') as f:
                b_data = f.read()
                data_combine += b_data
                # remove temp file
                os.remove(data_file_path)
        # uncompress
        print('-' * 50)
        decoded_data = zlib.decompress(bytes(data_combine))
        # get_file and data
        file_name = self.bytes_to_file_name(decoded_data[:256])
        # fix output folder
        if os.path.isdir(self.output):
            self.output = os.path.join(self.output, file_name)
        else:
            os.makedirs(os.path.dirname(self.output), exist_ok=True)
            self.output = os.path.join(os.path.dirname(self.output),
                                       os.path.split(self.output)[1] + os.path.splitext(file_name)[1])
        decoded_data = decoded_data[256:]
        # export
        with open(self.output, 'wb') as w:
            w.write(decoded_data)
        print('done ~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to image code; for screenshot')
    # argument for project
    parser.add_argument('--input', '-i', type=str, help='the source', default='demo')
    parser.add_argument('--output', '-o', type=str, help='the target', default=None)
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for image
    parser.add_argument('--level', '-l', type=int, help='the quality level of image: 1, 2, 3, 4', default=2)
    parser.add_argument('--image_size', '-s', type=int, help='the size of image to encode', nargs='+',
                        default=[800, 600])
    args = parser.parse_args()

    # fix output image size
    # if one number ==> square image
    if len(args.image_size) == 1:
        # noinspection PyUnresolvedReferences
        args.image_size = [args.image_size[0], args.image_size[0]]
    # check output image size
    if max(args.image_size) <= 3:
        raise ValueError('the image size will be larger than 3 pixels for each side')
    # check level number error
    if args.level > 4 or args.level < 1:
        raise ValueError('the level number should be 1, 2, 3 or 4')

    # encoding and decoding
    if os.path.exists(args.input) and os.path.isfile(args.input):
        if args.output is None:
            print(f'now encoding [{args.input}] ~')
        else:
            print(f'now encoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        f2img = File2Image(args)
        f2img.encode()
    elif os.path.exists(args.input) and os.path.isdir(args.input):
        if args.output is None:
            print(f'now decoding [{args.input}] ~')
        else:
            print(f'now decoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        img2f = Image2File(args)
        img2f.decode()
    else:
        raise TypeError('input not recognized')

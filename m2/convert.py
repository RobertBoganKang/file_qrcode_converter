import argparse
import multiprocessing as mp
import os
import readline
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
        * 7 pixels store the information of `index`,
            `level` for encoding and decoding process,
            and mask rgb channel info
            * `16_bits` stores the index [0, 65535].
            * `2_bits` stores the level [0, 3].
            * `3_bits` stores the rgb mask: 3 channels.
    ----------------------------------------------
    Description of header info of data (store file name):
        [256 Bytes store 256 characters] to store file name
    """

    def __init__(self):
        self.level = None
        self.image_size = None

        # initialize
        self.n_digits = None
        self.expand_n_digits = None
        self.n_base = None
        self.n_gap = None
        self.image_data_carry = None
        self.image_data_raw_size = None
        self.last_empty_data_carry = None
        self.rbk_array = None
        self.rgb_channel_mask_valid = None
        # black frame color defines here
        self.frame_color = [0, 0, 0]
        # image size limit defines here (2K screen size)
        self.image_size_limit = [1900, 1000]
        self.golden_ratio = (np.sqrt(5) - 1) / 2
        # rgb channel mask
        self.rgb_channel_mask = [0, 0, 0]
        # constants
        self.header_pixel_number = 7

        # command line fix
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")

    @staticmethod
    def error_print(content):
        raise ValueError(content)

    def retype_size(self, array_list):
        """ try to type new image size """
        while True:
            # noinspection PyBroadException
            try:
                new_size = input('--> please type a new image size (split with space): ')
                new_size = new_size.strip().split()
                if len(new_size) == 0:
                    self.error_print('please type at least one value')
                self.image_size = [int(x) for x in new_size]
            except Exception:
                print('warning: input not recognized!')
                continue
            break
        self.fix_image_parameters(array_list)

    def check_size_limit(self, num):
        if num == 0:
            pattern = '{-}'
        else:
            pattern = '{|}'
        if self.image_size[num] > self.image_size_limit[num]:
            print(
                f'warning: (image size)[{num}] @ {pattern} too big '
                f'({self.image_size[num]}px > {self.image_size_limit[num]}px)!'
            )
            return True
        else:
            return False

    def fix_image_parameters(self, array_list):
        """
        fix image parameters if image size parameters are not given for one or all
        @param array_list: np.array(int)
        @return: None
        """
        # check and fix image size
        # if image size is None, try to encode as one image first, then encode with default image size
        if self.image_size is None:
            self.image_size = [0, 0]
        # if one number ==> square image
        if len(self.image_size) == 1:
            # noinspection PyUnresolvedReferences
            self.image_size = [self.image_size[0], self.image_size[0]]
        # check output image size
        if 0 < self.image_size[0] * self.image_size[1] <= self.header_pixel_number + np.ceil(8 / 3):
            print('warning: the image size too small!')
            self.retype_size(array_list)

        retype = False
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            print('--> trying to convert into one image ~')
            # set default smallest image size mode
            if self.image_size[0] <= 0 and self.image_size[1] <= 0:
                # smallest image size mode, image with golden ratio
                size = np.sqrt(
                    (len(array_list) / self.rgb_channel_mask_valid + self.header_pixel_number) / self.golden_ratio)
                self.image_size[0] = int(np.ceil(size))
                self.image_size[1] = int(np.ceil(self.golden_ratio * size))
                if self.image_size[0] > self.image_size_limit[1]:
                    print(f'warning: failed to convert into one image!')
                    retype = True
            # fix one image size
            elif self.image_size[0] <= 0:
                self.image_size[0] = int(
                    np.ceil(
                        (len(array_list) / self.rgb_channel_mask_valid + self.header_pixel_number) / self.image_size[
                            1]))
                retype = self.check_size_limit(0) or retype
            elif self.image_size[1] <= 0:
                self.image_size[1] = int(
                    np.ceil(
                        (len(array_list) / self.rgb_channel_mask_valid + self.header_pixel_number) / self.image_size[
                            0]))
                retype = self.check_size_limit(1) or retype
        else:
            retype = self.check_size_limit(0) or retype
            retype = self.check_size_limit(1) or retype
        # initialize image
        self.initialize_image_size(image_size=self.image_size)
        # if retype, do it again
        if retype:
            self.retype_size(array_list)

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
        # given seven `1_bit` * `3-channels` pixels as header to store info
        # 3-channels * 7-pixels = 16_bit-index + 2_bit-level + 3_bit-channel_mask == 21_bits
        body_pixels = (self.image_size[0] * self.image_size[1] - self.header_pixel_number)
        bytes_to_encode = int(body_pixels * self.rgb_channel_mask_valid / self.n_digits)
        self.image_data_raw_size = body_pixels * 3
        # the number of data for each image to carry
        self.image_data_carry = bytes_to_encode * self.n_digits
        # crop it to store Bytes not bit splits (to make sure each image will decode complete bytes),
        # the rest will fill with empty white pixels, the number of last data to be empty
        self.last_empty_data_carry = (
                self.image_data_raw_size - int(self.image_data_carry / self.rgb_channel_mask_valid * 3))

    @staticmethod
    def cpu_count(cpu):
        """
        get the cpu number
        :return: int; valid cpu number
        """
        max_cpu = mp.cpu_count()
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
        self.compress = ops.compress
        # try image size and fix later
        self.image_size = ops.image_size

        # initialize with given parameters
        self.check_args_input(ops)
        self.initialize_level(level=ops.level)

        # fix input path
        self.check_output_folder()

        # rgb channel info
        self.rgb_channel_string = ops.channel_mask

    def check_args_input(self, ops):
        # check compression level
        if ops.compress > 9 or ops.compress < -1:
            self.error_print('the compression level should in the range of -1 ~ 9')
        # check level number error
        if ops.level > 4 or ops.level < 1:
            self.error_print('the level number should be 1, 2, 3 or 4')

    def initialize_channel_mask_info(self):
        string = self.rgb_channel_string
        """ extract rgb channel mask information into array """
        if 'r' in string:
            self.rgb_channel_mask[0] = 1
        if 'g' in string:
            self.rgb_channel_mask[1] = 1
        if 'b' in string:
            self.rgb_channel_mask[2] = 1
        self.rgb_channel_mask_valid = np.sum(self.rgb_channel_mask)
        if self.rgb_channel_mask_valid == 0:
            self.error_print('rgb channel mask should have at least one channel to be valid.')

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

    @staticmethod
    def chunk_it(seq, num):
        """
        split a list into n parts
        [https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length]
        :param seq: list(<>); array
        :param num: int; number of splits
        :return: array
        """
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

    def split_array_single(self, array, idx):
        """ process encoding of each chunk """
        array_list = []

        for a in array:
            array_list.extend(self.to_base(a, self.n_base, self.n_digits))
        return idx, array_list

    def split_array(self, array):
        """
        split the `1-Byte` or `8-bit` into several chunks of data
        @param array: list(int); range of integer is `0 ~ 255`
        """
        # split chunks
        array = self.chunk_it(array, self.cpu_number)
        pool = mp.Pool(self.cpu_number)
        results = []
        for i, arr in enumerate(array):
            results.append(pool.apply_async(self.split_array_single, args=(arr, i,)))
        pool.close()
        pool.join()

        results = [x.get() for x in results]
        results.sort(key=lambda x: x[0])
        array_list = []
        for res in results:
            array_list.extend(res[-1])

        return array_list

    def encode_header(self, i):
        result = self.to_base(i, 2, 16) + self.to_base(self.level - 1, 2, 2) + self.rgb_channel_mask
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
        return self.add_frame(image_reshaped, self.frame_color)

    def export_image_helper(self, image_array, i, out_folder):
        image_reshaped = self.array_transform(image_array)
        img = Image.fromarray(image_reshaped)
        out_image_path = os.path.join(out_folder, str(i) + '.png')
        img.save(out_image_path)
        print(f'[{out_image_path}] has been exported ~')

    def encode_pixel_with_mask(self, array_list, ii, total):
        rebuild_array = []
        # starting encoding
        # index of flatten data in pixel
        i_flatten_pix = 0
        while i_flatten_pix < total:
            mask_channel = self.rgb_channel_mask[i_flatten_pix % 3]
            if mask_channel == 1:
                # 1 to encode
                rebuild_array.append(self.data_to_pixel(array_list[ii]))
                ii += 1
            else:
                # 0 to fill useless pattern
                rebuild_array.append(self.rbk_array[i_flatten_pix + 21])
            i_flatten_pix += 1
        return rebuild_array

    @staticmethod
    def rbk_background(s1, s2):
        """
        generate `rbk` background with given image size
        @param s1: int; image size 0
        @param s2: int; image size 1
        @return: numpy array of image
        """
        # rbk one image pattern
        rbk = 'rrr..bbb..k..k.....b...k...,' \
              'r..r.b..b.k.k..r.r.bb..k.k.,' \
              'rrr..bbb..kk...rr..b.b.kk..,' \
              'r.r..b..b.k.k..r...b.b.k.k.,' \
              'r..r.bbb..k..k.r...bb..k.k.,' \
              '...........................'

        rbk = rbk.split(',')
        r_color = [255, 0, 0]
        # g_color = [0, 255, 0]
        b_color = [0, 0, 255]
        k_color = [64, 64, 64]
        dot_color = [255, 255, 255]
        rbk_size = (len(rbk), len(rbk[0]))
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
        table = np.array(table)
        return table

    def encode_image_single(self, bundles):
        array_list, out_folder, i = bundles
        image_array = np.array(self.encode_header(i)
                               + self.encode_pixel_with_mask(array_list, i * self.image_data_carry,
                                                             int(
                                                                 self.image_data_carry / self.rgb_channel_mask_valid * 3
                                                             ))
                               + [255 for _ in range(self.last_empty_data_carry)])
        self.export_image_helper(image_array, i, out_folder)

    def encode_image(self, array_list, out_folder):
        """
        split image
        @param array_list: list(int); data to encode
        @param out_folder: str; exported output folder
        @return: None
        """
        # fix image size
        self.fix_image_parameters(array_list)
        # full sized image
        i_length = len(array_list) // self.image_data_carry
        # if file too big for index
        if i_length > 65535:
            self.error_print('file is too big ~')
        bundles = [[array_list, out_folder, x] for x in range(i_length)]
        # extract rbk background
        self.rbk_array = self.rbk_background(self.image_size[0], self.image_size[1]).flatten()
        # encode images
        with mp.Pool(self.cpu_number) as pool:
            pool.map(self.encode_image_single, bundles)
        last_array_list = array_list[i_length * self.image_data_carry:]
        # encode image for rest of data
        if len(last_array_list) != 0:
            total_size = int(np.ceil(len(last_array_list) / self.rgb_channel_mask_valid * 3))
            # flatten the multi-dimensional table to array
            image_array = np.array(
                self.encode_header(i_length)
                + self.encode_pixel_with_mask(last_array_list, 0, total_size)
                # compress algorithm will not consider data at the end
                # final empty data with interesting background
                # add rbk background
                + [self.rbk_array[i] for i in
                   range(self.image_size[0] * self.image_size[1] * 3
                         - (self.image_data_raw_size - total_size),
                         self.image_size[0] * self.image_size[1] * 3
                         )
                   ]
            )
            self.export_image_helper(image_array, i_length, out_folder)

    def encode(self):
        self.initialize_channel_mask_info()
        with open(self.input, 'rb') as f:
            b_arr = f.read()
            # add file name
            name_bytes = self.path_to_bytes(self.input)
            # compress
            b_compress = zlib.compress(name_bytes + bytes(b_arr), self.compress)
            compressed_byte_list = list(b_compress)
            encoded_data = self.split_array(compressed_byte_list)
            os.makedirs(self.output, exist_ok=True)
            self.encode_image(encoded_data, self.output)


class SingleImage2TempFile(Common):
    """ decode one image to temp file """

    def __init__(self):
        super().__init__()
        self.image_number = None

    @staticmethod
    def from_base(integer_array, base):
        """ convert list of splits into 8-bit integer"""
        integer = 0
        for i in integer_array:
            integer *= base
            integer += i
        return int(integer)

    @staticmethod
    def remove_mask_array(array, rgb_channel_mask):
        if np.sum(rgb_channel_mask) == 3:
            return array
        else:
            array_rebuild = []
            for i in range(len(array)):
                if rgb_channel_mask[i % 3] == 1:
                    array_rebuild.append(array[i])
            return array_rebuild

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
        return self.from_base(array[:16], 2), self.from_base(array[16:18], 2) + 1, array[18:21]

    def zig_zag_traversal_find_upper_left_corner(self, matrix, identifier, tolerance):
        """
        [https://www.geeksforgeeks.org/print-matrix-zag-zag-fashion/]
        zig-zag traversal to find identifier for the first time
        -------------------------------------------------------------
        | o///............ |     | o156............ |
        | //X---------+... |     | 24X---------+... |
        | /.|ooooooooo|... | --> | 3.|ooooooooo|... |
        | ..+---------+... |     | ..+---------+... |
        | ................ |     | ................ |
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
        for i in range(rows - 1):
            for j in range(columns - 1):
                # test edge pixels
                total = i + j
                # get upper left corner pixel, red channel
                # if `image_number` < power(2, 15), the first red channel is 255, otherwise 0
                if self.image_number < 2 ** 15:
                    upper_left_corner_px_test = (255 - matrix[i + 1, j + 1, 0]) < tolerance
                else:
                    upper_left_corner_px_test = True
                # find result, check corner
                # noinspection PyChainedComparisons
                if (
                        np.mean(np.abs(matrix[i, j] - identifier)) < tolerance
                        and np.mean(np.abs(matrix[i + 1, j] - identifier)) < tolerance
                        and np.mean(np.abs(matrix[i, j + 1] - identifier)) < tolerance
                        and upper_left_corner_px_test
                ):
                    return i, j
                if total % 2 == 0:
                    # add at beginning
                    solution[total].insert(0, matrix[i, j])
                else:
                    # add at end of the list
                    solution[total].append(matrix[i, j])
        return None

    def find_bottom_right_corner_crop_image(self, matrix, target, idx, tolerance):
        """
        find the bottom-right pixel index from upper-left pixel index \
            and crop the image.
        -------------------------------------------------------------
        | ................ |
        | ..o>>>>>>>>>*... |
        | ..|oooooooooV... |
        | ..+---------X... |
        | ................ |
        -------------------------------------------------------------
        namely: from upper-left corner pixel `o` \
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
                else:
                    self.error_print('cannot find upper-right corner!')
        # find to the bottom
        while j < len(matrix[0]) - 1:
            if np.mean(np.abs(matrix[i, j + 1] - target)) < tolerance:
                j += 1
            else:
                # check corner
                if np.mean(np.abs(matrix[i - 1, j] - target)) < tolerance:
                    break
                else:
                    self.error_print('cannot find bottom-right corner!')
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

        # try values
        initial_tolerance = 8
        while True:
            if initial_tolerance > 64:
                self.error_print(f'[{path}] cannot be decoded!')
            # noinspection PyBroadException
            try:
                # try several tolerance to find corner of black frame
                # find upper-left corner
                index = self.zig_zag_traversal_find_upper_left_corner(img, self.frame_color, initial_tolerance)
                if index is not None:
                    # try several tolerance to find edges
                    cropped_image = self.find_bottom_right_corner_crop_image(img, self.frame_color, index,
                                                                             initial_tolerance * 2)
                    return cropped_image.flatten()
                else:
                    self.error_print('cannot find upper-left corner!')
            except Exception:
                initial_tolerance *= 2

    def path_decode_to_temp_data(self, path):
        # extract data from image
        data = self.image_path_to_data(path)
        index, level, rgb_channel_mask = self.decode_header(data[:21])
        # get all parameters for decoding
        self.initialize_level(level)
        file_data = [self.pixel_to_data(x) for x in data[21:]]
        file_data_no_mask = self.remove_mask_array(file_data, rgb_channel_mask)
        decoded_data = self.combine_array(file_data_no_mask)
        return index, decoded_data

    def decode_to_temp_file(self, params):
        path, self.image_number = params
        index, decoded_data = self.path_decode_to_temp_data(path)
        # export data file
        out_path = os.path.join(os.path.dirname(path), str(index) + '.tmp')
        with open(out_path, 'wb') as w:
            w.write(bytes(decoded_data))
        print(f'[{path}] has been decoded ~')


class Image2File(Common):
    """ decoder """

    def __init__(self, ops):
        super().__init__()
        self.input = ops.input
        self.output = self.output = Common.fix_out_path(self.input, ops.output, encode=False)
        self.cpu_number = Common.cpu_count(ops.cpu_number)

        self.image_number = None

    @staticmethod
    def bytes_to_file_name(b):
        d = [x for x in b if x != 0]
        return bytes(d).decode('utf-8')

    def decode_image_to_temp_file_single(self, path):
        # create new object for each image of decoding
        decode = SingleImage2TempFile()
        decode.decode_to_temp_file([path, self.image_number])

    def decode_image_to_temp_file(self):
        image_files = os.listdir(self.input)
        image_files = [os.path.join(self.input, x) for x in image_files if not x.endswith('.tmp')]
        # if no image found
        if len(image_files) == 0:
            self.error_print('no image found!')
        # get number of images
        self.image_number = len(image_files)
        # decode images
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
                self.error_print(f'[{data_file_path}] data file not found!')
            with open(data_file_path, 'rb') as f:
                b_data = f.read()
                data_combine += b_data
            # remove temp file
            os.remove(data_file_path)
        # uncompress
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to image code; for screenshot')
    # argument for project
    parser.add_argument('--input', '-i', type=str, help='the source', default=None)
    parser.add_argument('--output', '-o', type=str, help='the target', default=None)
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for image
    parser.add_argument('--level', '-l', type=int, help='the quality level of image: 1, 2, 3, 4', default=1)
    parser.add_argument('--channel_mask', '-c', type=str, help='channel mask for image, '
                                                               'write `r`, `g`, `b` or its combinations',
                        default='rgb')
    parser.add_argument('--compress', '-z', type=int, help='the encoding compression level: 0 ~ 9 or -1 as default',
                        default=1)
    parser.add_argument('--image_size', '-s', type=int,
                        help='the size of image to encode (pixel); '
                             'if given one image size, the image would be square; '
                             'if one or all parameter are `<=0`, it will convert into one image '
                             'but to consider this image size are not given.', nargs='+', default=None)
    args = parser.parse_args()

    # encoding and decoding
    if args.input is not None and os.path.exists(args.input) and os.path.isfile(args.input):
        if args.output is None:
            print(f'now encoding [{args.input}] ~')
        else:
            print(f'now encoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        f2img = File2Image(args)
        f2img.encode()
    elif args.input is not None and os.path.exists(args.input) and os.path.isdir(args.input):
        if args.output is None:
            print(f'now decoding [{args.input}] ~')
        else:
            print(f'now decoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        img2f = Image2File(args)
        img2f.decode()
    else:
        raise FileNotFoundError('input not recognized')

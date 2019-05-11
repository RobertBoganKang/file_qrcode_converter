import argparse
import base64
import multiprocessing
import os
import shutil
import zlib

from PIL import Image


################
# common utils #
################

class CommonUtils(object):
    def __init__(self, ops):
        # the number of byte to store index
        self.idx_byte = ops.index_byte
        # color space (CMYK does not work)
        self.color_space = 'RGB'
        self.color_dim = len(self.color_space)
        self.out_format = self.determine_format()

    @staticmethod
    def cpu_count(cpu_count):
        """
        get the cpu number
        :return: int; valid cpu number
        """
        max_cpu = multiprocessing.cpu_count()
        if cpu_count == 0 or cpu_count > max_cpu:
            cpu_count = max_cpu
        return cpu_count

    def determine_format(self):
        if self.color_space.lower() != 'rgb':
            return '.tif'
        else:
            return '.png'


class DecoderUtil(CommonUtils):
    def __init__(self, ops):
        super().__init__(ops)

    @staticmethod
    def byte_to_index(index_array):
        """
        convert byte array to index
        """
        i = 0
        index = 0
        while i < len(index_array):
            index *= 256
            index += index_array[i]
            i += 1
        return index

    @staticmethod
    def string_to_byte_array(data):
        return zlib.decompress(base64.b64decode(data))

    def get_data(self, data):
        # decode letters to bytes
        data = self.string_to_byte_array(data)
        idx = self.byte_to_index(data[:self.idx_byte])
        if idx == 0:
            data_effective = data[2 * self.idx_byte:]
            max_chunk.value = self.byte_to_index(data[self.idx_byte:2 * self.idx_byte])
        else:
            data_effective = data[self.idx_byte:]
        return data_effective, idx


class EncoderUtil(CommonUtils):
    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.cpu_number = self.cpu_count(ops.cpu_number)

    @staticmethod
    def index_to_byte(index):
        """
        convert index to byte array
        :param index: int; int < 2^16
        :return: list(int); int < 2^8
        """
        index_array = []
        while index > 0:
            index_array.append(index % 256)
            index = index // 256
        if len(index_array) == 1:
            index_array.append(0)
        index_array.reverse()
        return index_array

    @staticmethod
    def byte_array_to_string(b):
        return base64.b64encode(zlib.compress(bytes(b)))

    def create_chunks(self):
        with open(self.input, 'rb') as f:
            byte_array = f.read()
        byte_array = zlib.compress(byte_array)
        i = 0
        file_counter = 0
        bucket = [0 for _ in range(self.idx_byte * 2)]
        chunks = []
        while i < len(byte_array):
            # check chunk length not to exceed 2^(8*idx_byte) files
            if file_counter >= 2 ** (self.idx_byte * 8):
                raise OverflowError('number of images is [{}] > {}!'.format(len(chunks), 2 ** (self.idx_byte * 8)))
            if (i + self.idx_byte) % (self.chunk_size - self.idx_byte) == 0:
                chunks.append(self.index_to_byte(file_counter) + bucket)
                bucket = []
                file_counter += 1
            bucket.append(byte_array[i])
            i += 1
        if len(bucket) != 0:
            chunks.append(self.index_to_byte(file_counter) + bucket)

        # make up total chunk size marker
        chunk_size = self.index_to_byte(len(chunks))
        chunks[0][self.idx_byte:2 * self.idx_byte] = chunk_size
        return chunks


############################
# QR-Code & File Converter #
############################

class QR2File(DecoderUtil):
    """ decode """

    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = ops.output
        self.cpu_number = self.cpu_count(ops.cpu_number)

    # separate channels of image
    def separate_image(self, img_path):
        name = os.path.splitext(os.path.split(img_path)[1])[0]
        color_channels = Image.open(img_path).convert(self.color_space).split()
        for i in range(self.color_dim):
            color_channels[i].save(os.path.join(self.input, '#_' + str(name) + '_' + self.color_space[i] + '.png'))

    def separate_all_image(self):
        print('separating image channels ~')
        names_ = os.listdir(self.input)
        names = []
        # select and clean all other files
        for x in names_:
            if not x.startswith('#_') and '.' in x:
                names.append(x)
            else:
                os.remove(os.path.join(self.input, x))
        fs = []
        for name in names:
            fs.append(os.path.join(self.input, name))
        if len(fs) == 0:
            raise FileNotFoundError('no image found!')
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.separate_image, fs)

    # decode data
    @staticmethod
    def try_size_to_decode(path):
        from pyzbar.pyzbar import decode
        """ try different sizes to decode """
        img = Image.open(path)
        for size in range(300, 1500, 100):
            imgc = img.copy()
            img_size = list(imgc.size)
            img_size = [size, int(img_size[1] / img_size[0] * size)]
            imgc = imgc.resize(img_size)
            data = decode(imgc)
            if len(data) == 0:
                continue
            else:
                return data
        print('[{}] cannot be decoded, please replace it with new image!'.format(path))
        return None

    def extract_helper(self, f):
        """ extract bytes """
        in_path = os.path.join(self.input, f)
        data = self.try_size_to_decode(in_path)
        if data is None:
            return
        data = data[0].data
        data_effective, idx = self.get_data(data)
        print('[{}] has been analyzed ~'.format(f))
        export_path = os.path.join(self.input, str(idx))
        if not os.path.exists(export_path):
            with open(export_path, 'wb') as w:
                w.write(data_effective)
        os.remove(in_path)

    def extract_data_from_qr(self):
        """ main function for extraction"""
        global max_chunk
        max_chunk = multiprocessing.Value('i')
        pool = multiprocessing.Pool(self.cpu_number)
        fs = os.listdir(self.input)
        fs = [x for x in fs if x.startswith('#_')]
        pool.map(self.extract_helper, fs)

        data_rebuild = b''
        for i in range(max_chunk.value):
            data_file_path = os.path.join(self.input, str(i))
            with open(data_file_path, 'rb') as f:
                image_byte_array = f.read()
                data_rebuild += image_byte_array
            os.remove(data_file_path)
        data_rebuild = zlib.decompress(data_rebuild)

        print('-' * 50)
        with open(self.output, 'wb') as w:
            w.write(data_rebuild)
        print('[{}] has been exported ~'.format(self.output))


class File2QR(EncoderUtil):
    """ encode """

    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.quality = ops.quality
        self.cpu_number = self.cpu_count(ops.cpu_number)
        self.check_output_folder()

    def ask_for_new_path(self):
        new_path = input('please type new path for exporting qr-code:')
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
                    check = input('the folder is not empty, do you with to remove them [y/n]: ')
                    if check == '':
                        continue
                    if check.strip().lower() in ['yes', 'yeah', 'yep', 'y']:
                        shutil.rmtree(self.output)
                        os.makedirs(self.output)
                        return
                    elif check.strip().lower() in ['no', 'n']:
                        self.ask_for_new_path()
                        self.check_output_folder()
                        return
                    else:
                        print('input not recognized!')

    def prepare_qr_code_image(self, data, out_path):
        import pyqrcode
        # require `pypng` library
        img = pyqrcode.create(data, error=self.quality)
        img.png(out_path, scale=10)

    def export_qr_code_helper(self, chunk):
        i, chunk = chunk
        compressed = self.byte_array_to_string(chunk)
        path = os.path.join(self.output, 't' + str(i) + '.png')
        self.prepare_qr_code_image(compressed, path)
        print('[{}] has been export successfully ~'.format(path))

    def export_qr_code(self):
        chunks_raw = self.create_chunks()
        chunks_raw = [(i, x) for i, x in enumerate(chunks_raw)]
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.export_qr_code_helper, chunks_raw)

    # combine image into image channels
    def merge_image(self, f):
        img_arr, out_path = f
        color_channels = [Image.open(x).convert('L') for x in img_arr]
        # size check
        sizes = [x.size for x in color_channels]
        is_size_equal = True
        for i in range(len(sizes) - 1):
            is_size_equal = is_size_equal and sizes[i] == sizes[i + 1]
        if not is_size_equal:
            color_channels = [x.resize(sizes[0]) for x in color_channels]
        new_image = Image.merge(self.color_space, color_channels)
        new_image.save(out_path)
        # remove used image
        for i in img_arr:
            os.remove(i)
        print('[{}] has been merged successfully ~'.format(out_path))

    def convert_image_color_space(self, in_path, out_path):
        img = Image.open(in_path).convert('L')
        color_channel = [img for _ in range(self.color_dim)]
        new_img = Image.merge(self.color_space, color_channel)
        new_img.save(out_path)
        os.remove(in_path)

    def replace_with_merged_image(self):
        print('-' * 50)
        names = os.listdir(self.output)
        names = [x for x in names if x.startswith('t')]
        length = (len(names) - 1) // self.color_dim
        fs = []
        if length > 0:
            for i in range(length):
                out_path = os.path.join(self.output, str(i) + self.out_format)
                img_array = [os.path.join(self.output, 't' + str(x) + '.png') for x in
                             range(i * self.color_dim, (i + 1) * self.color_dim)]
                fs.append((img_array, out_path))
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.merge_image, fs)

        # change name of last few images
        if length * self.color_dim < len(names):
            i = length
            for j in range(length * self.color_dim, len(names)):
                in_path = os.path.join(self.output, 't' + str(j) + '.png')
                out_path = os.path.join(self.output, str(i) + self.out_format)
                self.convert_image_color_space(in_path, out_path)
                print('[{}] color has been converted successfully ~'.format(out_path))
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to qr code')
    # argument for project
    parser.add_argument('--input', '-i', type=str, help='the source', default='demo')
    parser.add_argument('--output', '-o', type=str, help='the target', default='mozart_11.mid')
    parser.add_argument('--chunk_size', '-s', type=int, help='chunk size to encode', default=2048)
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for qr-code
    parser.add_argument('--quality', '-q', type=str, help='the quality of qr-code: L, M, Q, H', default='L')
    parser.add_argument('--black_white', '-bw', action='store_true', help='the black white mode for qr-code if have')

    # encode control (Danger!!)
    parser.add_argument('--index_byte', '-x', type=int, help='the byte of index info have been used', default=2)

    args = parser.parse_args()

    if os.path.exists(args.input) and os.path.isfile(args.input):
        print('now encoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        f2qr = File2QR(args)
        f2qr.export_qr_code()
        if not args.black_white:
            f2qr.replace_with_merged_image()
    elif os.path.exists(args.input) and os.path.isdir(args.input):
        print('now decoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        qr2f = QR2File(args)
        if not args.black_white:
            qr2f.separate_all_image()
        qr2f.extract_data_from_qr()
    else:
        raise TypeError('input not recognized')
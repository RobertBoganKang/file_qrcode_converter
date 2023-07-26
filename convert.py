import argparse
import base64
import multiprocessing as mp
import os
import readline
import shutil
import zlib

from PIL import Image


class QRCommon(object):
    def __init__(self, ops):
        # the number of byte to store index
        self.idx_byte = ops.index_byte
        # color space (CMYK does not work)
        self.color_space = 'RGB'
        self.color_dim = len(self.color_space)
        self.out_format = self.determine_format()

        # command line fix
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")

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

    def determine_format(self):
        if self.color_space.lower() != 'rgb':
            return '.tif'
        else:
            return '.png'

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


class QRDecoder(QRCommon):
    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = self.fix_out_path(self.input, ops.output, encode=False)
        self.max_chunk = mp.Manager().list()

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
            self.max_chunk.append(self.byte_to_index(data[self.idx_byte:2 * self.idx_byte]))
        else:
            data_effective = data[self.idx_byte:]
        return data_effective, idx

    @staticmethod
    def bytes_to_file_name(b):
        d = [x for x in b if x != 0]
        return bytes(d).decode('utf-8')


class QREncoder(QRCommon):
    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = self.fix_out_path(self.input, ops.output)
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
    def path_to_bytes(path):
        path = os.path.split(path)[1]
        b = path.encode('utf-8')
        for _ in range(256 - len(b)):
            b += b'\x00'
        return b

    @staticmethod
    def byte_array_to_string(b):
        return base64.b64encode(zlib.compress(bytes(b), 9))

    def create_chunks(self):
        with open(self.input, 'rb') as f:
            byte_array = f.read()
        # add file name here
        byte_array = self.path_to_bytes(self.input) + byte_array
        byte_array = zlib.compress(byte_array, 9)
        i = 0
        file_counter = 0
        bucket = [0 for _ in range(self.idx_byte * 2)]
        chunks = []
        while i < len(byte_array):
            # check chunk length not to exceed 2^(8*idx_byte) files
            if file_counter >= 2 ** (self.idx_byte * 8):
                raise OverflowError(f'ERROR: number of images is [{len(chunks)}] > {2 ** (self.idx_byte * 8)}!')
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


class QR2File(QRDecoder):
    """ decode """

    def __init__(self, ops):
        super().__init__(ops)
        self.cpu_number = self.cpu_count(ops.cpu_number)
        self.black_white = ops.black_white

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
                path = os.path.join(self.input, x)
                os.remove(path)
        fs = []
        for name in names:
            fs.append(os.path.join(self.input, name))
        if len(fs) == 0:
            raise FileNotFoundError('ERROR: no image found!')
        with mp.Pool(self.cpu_number) as pool:
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
        print(f'[{path}] cannot be decoded, please replace it with new image!')
        return None

    def extract_helper(self, f):
        """ extract bytes """
        in_path = os.path.join(self.input, f)
        data = self.try_size_to_decode(in_path)
        if data is None:
            return
        data = data[0].data
        data_effective, idx = self.get_data(data)
        print(f'[{f}] has been analyzed ~')
        export_path = os.path.join(self.input, str(idx))
        if not os.path.exists(export_path):
            with open(export_path, 'wb') as w:
                w.write(data_effective)
        if not self.black_white:
            os.remove(in_path)

    def extract_data_from_qr(self):
        """ main function for extraction"""
        # noinspection PyGlobalUndefined
        with mp.Pool(self.cpu_number) as pool:
            names = []
            names_ = os.listdir(self.input)
            # select and clean all other files
            if self.black_white:
                for x in names_:
                    if x.startswith('#_') or '.' not in x:
                        path = os.path.join(self.input, x)
                        os.remove(path)
                    elif '.' in x:
                        names.append(x)
            else:
                names = [x for x in names_ if x.startswith('#_')]
            pool.map(self.extract_helper, names)

        if len(self.max_chunk) == 0:
            raise ValueError('ERROR: first image missing')

        data_rebuild = b''
        for i in range(self.max_chunk[0]):
            data_file_path = os.path.join(self.input, str(i))
            with open(data_file_path, 'rb') as f:
                image_byte_array = f.read()
                data_rebuild += image_byte_array
            os.remove(data_file_path)
        data_rebuild = zlib.decompress(data_rebuild)

        print('-' * 50)
        # extract file name
        file_name = self.bytes_to_file_name(data_rebuild[:256])
        data_rebuild = data_rebuild[256:]
        # fix output folder
        if os.path.isdir(self.output):
            self.output = os.path.join(self.output, file_name)
        else:
            os.makedirs(os.path.dirname(self.output), exist_ok=True)
            self.output = os.path.join(os.path.dirname(self.output),
                                       os.path.split(self.output)[1] + os.path.splitext(file_name)[1])
        with open(self.output, 'wb') as w:
            w.write(data_rebuild)
        print(f'[{self.output}] has been exported ~')


class File2QR(QREncoder):
    """ encode """

    def __init__(self, ops):
        super().__init__(ops)
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

    def prepare_qr_code_image(self, data, out_path):
        import segno
        # require `pypng` library
        img = segno.make_qr(data, error=self.quality)
        img.save(out_path, scale=10)

    def export_qr_code_helper(self, chunk):
        i, chunk = chunk
        compressed = self.byte_array_to_string(chunk)
        path = os.path.join(self.output, 't' + str(i) + '.png')
        self.prepare_qr_code_image(compressed, path)
        print(f'[{path}] has been export successfully ~')

    def export_qr_code(self):
        chunks_raw = self.create_chunks()
        chunks_raw = [(i, x) for i, x in enumerate(chunks_raw)]
        with mp.Pool(self.cpu_number) as pool:
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
        print(f'[{out_path}] has been merged successfully ~')

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
        with mp.Pool(self.cpu_number) as pool:
            pool.map(self.merge_image, fs)

        # change name of last few images
        if length * self.color_dim < len(names):
            i = length
            for j in range(length * self.color_dim, len(names)):
                in_path = os.path.join(self.output, 't' + str(j) + '.png')
                out_path = os.path.join(self.output, str(i) + self.out_format)
                self.convert_image_color_space(in_path, out_path)
                print(f'[{out_path}] color has been converted successfully ~')
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to qr code; for camera')
    # argument for project
    io_group = parser.add_argument_group('file i/o')
    io_group.add_argument('--input', '-i', type=str, help='the source', default=None)
    io_group.add_argument('--output', '-o', type=str, help='the target', default=None)
    io_group.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for qr-code
    qr_group = parser.add_argument_group('qr code arguments')
    qr_group.add_argument('--chunk_size', '-s', type=int, help='chunk size to encode', default=2048)
    qr_group.add_argument('--quality', '-q', type=str, help='the quality of qr-code: L, M, Q, H', default='L')
    qr_group.add_argument('--black_white', '-bw', action='store_true', help='the black white mode for qr-code if have')

    # encode control (Danger!!)
    qr_group.add_argument('--index_byte', '-x', type=int, help='the byte of index info have been used', default=2)

    args = parser.parse_args()

    # encode or decode
    if args.input is not None and os.path.exists(args.input) and os.path.isfile(args.input):
        if args.output is None:
            print(f'now encoding [{args.input}] ~')
        else:
            print(f'now encoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        f2qr = File2QR(args)
        f2qr.export_qr_code()
        if not args.black_white:
            f2qr.replace_with_merged_image()
    elif args.input is not None and os.path.exists(args.input) and os.path.isdir(args.input):
        if args.output is None:
            print(f'now decoding [{args.input}] ~')
        else:
            print(f'now decoding from [{args.input}] to [{args.output}] ~')
        print('-' * 50)
        qr2f = QR2File(args)
        if not args.black_white:
            qr2f.separate_all_image()
        qr2f.extract_data_from_qr()
    else:
        raise FileNotFoundError('ERROR: input not recognized')

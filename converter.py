import argparse
import base64
import multiprocessing
import os
import zlib

import pyqrcode
from PIL import Image
from pyzbar.pyzbar import decode


class CommonUtils(object):
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


class QR2File(CommonUtils):
    def __init__(self, ops):
        self.input = ops.input
        self.output = ops.output
        self.cpu_number = self.cpu_count(ops.cpu_number)

    @staticmethod
    def try_size_to_decode(path):
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
        raise RuntimeError('{} cannot be decoded, please replace it with new image!'.format(path))

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

    def extract_helper(self, f):
        """ extract bytes """
        data = self.try_size_to_decode(os.path.join(self.input, f))
        data = data[0].data
        # decode letters to bytes
        data = zlib.decompress(base64.b64decode(data))
        print('[{}] has been analyzed ~'.format(f))
        idx = self.byte_to_index(data[:2])
        if idx == 0:
            data_effective = data[4:]
            max_chunk.value = self.byte_to_index(data[2:4])
        else:
            data_effective = data[2:]
        with open(os.path.join(self.input, str(idx)), 'wb') as w:
            w.write(data_effective)

    def extract_data_from_qr(self):
        """ main function for extraction"""
        global max_chunk
        max_chunk = multiprocessing.Value('i')
        pool = multiprocessing.Pool(self.cpu_number)
        fs = os.listdir(self.input)
        fs = [x for x in fs if '.' in x]
        pool.map(self.extract_helper, fs)

        data_rebuild = b''
        for i in range(max_chunk.value):
            data_file_path = os.path.join(self.input, str(i))
            with open(data_file_path, 'rb') as f:
                image_byte_array = f.read()
                data_rebuild += image_byte_array
            os.remove(data_file_path)

        print('-' * 50)
        with open(self.output, 'wb') as w:
            w.write(data_rebuild)
        print('[{}] has been exported ~'.format(self.output))


class File2QR(CommonUtils):
    def __init__(self, ops):
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.quality = ops.quality
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
        i = 0
        file_counter = 0
        bucket = [0, 0, 0, 0]
        chunks = []
        while i < len(byte_array):
            if (i + 2) % (self.chunk_size - 2) == 0:
                chunks.append(self.index_to_byte(file_counter) + bucket)
                bucket = []
                file_counter += 1
            bucket.append(byte_array[i])
            i += 1
        if len(bucket) != 0:
            chunks.append(self.index_to_byte(file_counter) + bucket)

        # check chunk size not to exceed 65536 files
        if len(chunks) > 2 ** 16:
            raise OverflowError('number of images is [{}] > 65536!'.format(len(chunks)))

        # make up total chunk size marker
        chunk_size = self.index_to_byte(len(chunks))
        chunks[0][2:4] = chunk_size
        return chunks

    def prepare_qr_code_image(self, data, out_path):
        img = pyqrcode.create(data, error=self.quality)
        img.svg(out_path, scale=10)

    def export_qr_code_helper(self, chunk):
        i, chunk = chunk
        compressed = self.byte_array_to_string(chunk)
        path = os.path.join(self.output, str(i) + '.svg')
        self.prepare_qr_code_image(compressed, path)
        print('[{}] has been export successfully ~'.format(path))

    def export_qr_code(self):
        chunks_raw = self.create_chunks()
        chunks_raw = [(i, x) for i, x in enumerate(chunks_raw)]
        os.makedirs(self.output, exist_ok=True)
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.export_qr_code_helper, chunks_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to qr code')
    parser.add_argument('--input', '-i', type=str, help='the source', default='demo')
    parser.add_argument('--output', '-o', type=str, help='the target', default='mozart_11.mid')
    parser.add_argument('--chunk_size', '-s', type=int, help='chunk size to encode', default=2048)
    parser.add_argument('--quality', '-q', type=str, help='the quality of qr-code: L, M, Q, H', default='L')
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)
    args = parser.parse_args()

    if os.path.exists(args.input) and os.path.isfile(args.input):
        print('now encoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        f2qr = File2QR(args)
        f2qr.export_qr_code()
    elif os.path.exists(args.input) and os.path.isdir(args.input):
        print('now decoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        qr2f = QR2File(args)
        qr2f.extract_data_from_qr()
    else:
        raise TypeError('input not recognized')

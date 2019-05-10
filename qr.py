import argparse
import multiprocessing
import os

from PIL import Image

from utils import File2Code, Code2File


class QR2File(Code2File):
    def __init__(self, ops):
        self.input = ops.input
        self.output = ops.output
        self.cpu_number = self.cpu_count(ops.cpu_number)

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
        raise RuntimeError('{} cannot be decoded, please replace it with new image!'.format(path))

    def extract_helper(self, f):
        """ extract bytes """
        data = self.try_size_to_decode(os.path.join(self.input, f))
        data = data[0].data
        data_effective, idx = self.get_data(data, max_chunk)
        print('[{}] has been analyzed ~'.format(f))
        with open(os.path.join(self.input, str(idx)), 'wb') as w:
            w.write(data_effective)

    def extract_data_from_qr(self):
        """ main function for extraction"""
        global max_chunk
        max_chunk = multiprocessing.Value('i')
        pool = multiprocessing.Pool(self.cpu_number)
        fs = os.listdir(self.input)
        fs = [x for x in fs if '.' in x]
        if len(fs) == 0:
            raise FileNotFoundError('no image found!')
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


class File2QR(File2Code):
    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.quality = ops.quality
        self.cpu_number = self.cpu_count(ops.cpu_number)

    def prepare_qr_code_image(self, data, out_path):
        import pyqrcode
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
    # argument for project
    parser.add_argument('--input', '-i', type=str, help='the source', default='demo')
    parser.add_argument('--output', '-o', type=str, help='the target', default='mozart_11.mid')
    parser.add_argument('--chunk_size', '-s', type=int, help='chunk size to encode', default=2048)
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for qr-code
    parser.add_argument('--quality', '-q', type=str, help='the quality of qr-code: L, M, Q, H', default='L')
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

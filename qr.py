import argparse
import multiprocessing
import os
import shutil
import zlib

from PIL import Image

from utils import File2Code, Code2File


class QR2File(Code2File):
    """ decode """

    def __init__(self, ops):
        super().__init__()
        self.input = ops.input
        self.output = ops.output
        self.cpu_number = self.cpu_count(ops.cpu_number)

    # separate channels of image
    def separate_image(self, img_path):
        name = os.path.splitext(os.path.split(img_path)[1])[0]
        r, g, b = Image.open(img_path).convert('RGB').split()
        r.save(os.path.join(self.input, '#_' + str(name) + '_r.png'))
        g.save(os.path.join(self.input, '#_' + str(name) + '_g.png'))
        b.save(os.path.join(self.input, '#_' + str(name) + '_b.png'))

    def separate_all_image(self):
        print('separating image channels ~')
        names = os.listdir(self.input)
        names = [x for x in names if not x.startswith('#') and '.' in x]
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
        raise RuntimeError('{} cannot be decoded, please replace it with new image!'.format(path))

    def extract_helper(self, f):
        """ extract bytes """
        in_path = os.path.join(self.input, f)
        data = self.try_size_to_decode(in_path)
        data = data[0].data
        data_effective, idx = self.get_data(data, max_chunk)
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
        fs = [x for x in fs if x.startswith('#')]
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


class File2QR(File2Code):
    """ encode """

    def __init__(self, ops):
        super().__init__(ops)
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.quality = ops.quality
        self.cpu_number = self.cpu_count(ops.cpu_number)

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
        os.makedirs(self.output, exist_ok=True)
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.export_qr_code_helper, chunks_raw)

    # combine image into rgb channels
    @staticmethod
    def merge_image(img_arr, out_path):
        rgb = [Image.open(x).convert('L') for x in img_arr]
        new_image = Image.merge('RGB', rgb)
        new_image.save(out_path)
        # remove used image
        for i in img_arr:
            os.remove(i)

    def replace_with_merged_image(self):
        print('-' * 50)
        names = os.listdir(self.output)
        names = [x for x in names if x.startswith('t')]
        length = (len(names) - 1) // 3
        if length > 0:
            for i in range(length):
                out_path = os.path.join(self.output, str(i) + '.png')
                img_array = [os.path.join(self.output, 't' + str(x) + '.png') for x in range(i * 3, (i + 1) * 3)]
                self.merge_image(img_array, out_path)
                print('[{}] has been merged successfully ~'.format(out_path))
        # change name of last few images
        if length * 3 < len(names):
            i = length
            for j in range(length * 3, len(names)):
                out_path = os.path.join(self.output, str(i) + '.png')
                shutil.move(os.path.join(self.output, 't' + str(j) + '.png'), out_path)
                print('[{}] name has been changed successfully ~'.format(out_path))
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

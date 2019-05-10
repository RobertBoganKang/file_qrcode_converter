import argparse
import multiprocessing
import os
import subprocess

from utils import File2Code, Code2File


class JabCommon(object):
    def __init__(self):
        self.FILE = os.path.abspath(os.path.dirname(__file__))
        # set the working directory
        os.chdir(self.FILE)
        # the app for jabcode reader and writer
        self.jab_reader = os.path.join(self.FILE, 'src', 'jabcodeReader', 'bin', 'jabcodeReader')
        self.jab_writer = os.path.join(self.FILE, 'src', 'jabcodeWriter', 'bin', 'jabcodeWriter')

        # detect jabcode library
        self.jab_function_detect()

    def jab_function_detect(self):
        if not os.path.exists(self.jab_reader) or not os.path.exists(self.jab_writer):
            raise FileNotFoundError("""
            -------------------------------------------------------------------------
            instruction:
            1. please download the `jabcode` at
            -->[https://github.com/jabcode/jabcode].
            2. put the `src` folder on the root directory of this project.
            3. follow the instructions and compile all files.
            -------------------------------------------------------------------------
            troubleshooting:
            if `libpng16.a` or `libz.a` breaks [at `./src/jabcode/bin/*`]
            1. please download them at their `official website(s)`:
            -->[http://www.libpng.org and https://www.zlib.net/] .
            2. compile it, and find the corresponding file.
            3. replace them.
            -------------------------------------------------------------------------
            """)


class Jab2File(Code2File, JabCommon):
    def __init__(self, ops):
        Code2File.__init__(self)
        JabCommon.__init__(self)
        self.input = ops.input
        self.output = ops.output
        self.format_support = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        self.cpu_number = self.cpu_count(ops.cpu_number)

    def decode(self, path):
        # image convert if necessary
        path_split = os.path.splitext(path)
        is_png = False
        if path_split[1].lower() == '.png':
            is_png = True
        elif path_split[1].lower() in self.format_support:
            from PIL import Image
            img = Image.open(path)
            # modify path and export
            path = path_split[0] + '.png'
            img.save(path)
        else:
            return
        # decode
        out_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.split(path)[1])[0])
        subprocess.run([self.jab_reader, path, '--output', out_path])
        with open(out_path, 'r') as f:
            data = f.read()
        os.remove(out_path)
        if not is_png:
            os.remove(path)
        return data

    def extract_helper(self, f):
        """ extract bytes """
        data = self.decode(os.path.join(self.input, f))
        data_effective, idx = self.get_data(data, max_chunk)
        print('[{}] has been analyzed ~'.format(f))
        with open(os.path.join(self.input, str(idx)), 'wb') as w:
            w.write(data_effective)

    def extract_data_from_jab(self):
        """ main function for extraction"""
        global max_chunk
        max_chunk = multiprocessing.Value('i')
        pool = multiprocessing.Pool(self.cpu_number)
        fs = os.listdir(self.input)
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


class File2Jab(File2Code, JabCommon):
    def __init__(self, ops):
        File2Code.__init__(self, ops)
        JabCommon.__init__(self)
        self.input = ops.input
        self.output = ops.output
        self.chunk_size = ops.chunk_size
        self.image_size = str(ops.image_size)
        self.color_number = str(ops.color_number)
        self.quality = str(ops.quality)
        self.cpu_number = self.cpu_count(ops.cpu_number)

    def export_jab_code_image(self, data, out_path):
        subprocess.run(
            [self.jab_writer, '--input', data, '--output', out_path, '--symbol-number', '1', '--symbol-position', '0',
             '--symbol-version', self.image_size, self.image_size, '--color-number', self.color_number, '--ecc-level',
             self.quality])

    def export_jab_code_helper(self, chunk):
        i, chunk = chunk
        compressed = self.byte_array_to_string(chunk)
        path = os.path.join(self.output, str(i) + '.png')
        self.export_jab_code_image(compressed, path)
        if os.path.exists(path):
            print('[{}] has been export successfully ~'.format(path))

    def export_jab_code(self):
        chunks_raw = self.create_chunks()
        chunks_raw = [(i, x) for i, x in enumerate(chunks_raw)]
        os.makedirs(self.output, exist_ok=True)
        pool = multiprocessing.Pool(self.cpu_number)
        pool.map(self.export_jab_code_helper, chunks_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert file to qr code')
    # argument for project
    parser.add_argument('--input', '-i', type=str, help='the source', default='demoj')
    parser.add_argument('--output', '-o', type=str, help='the target', default='mozart_11.mid')
    parser.add_argument('--chunk_size', '-s', type=int, help='chunk size to encode', default=2048)
    parser.add_argument('--cpu_number', '-j', type=int, help='cpu number to process', default=0)

    # argument for jabcode
    parser.add_argument('--image_size', '-is', type=int, help='the size of image (1~32)', default=20)
    parser.add_argument('--color_number', '-ic', type=int, help='the color number of image 2^[2~8]', default=8)
    parser.add_argument('--quality', '-iq', type=int, help='the quality of jabcode (1~10)', default=3)

    args = parser.parse_args()

    if os.path.exists(args.input) and os.path.isfile(args.input):
        print('now encoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        f2qr = File2Jab(args)
        f2qr.export_jab_code()
    elif os.path.exists(args.input) and os.path.isdir(args.input):
        print('now decoding from [{}] to [{}] ~'.format(args.input, args.output))
        print('-' * 50)
        qr2f = Jab2File(args)
        qr2f.extract_data_from_jab()
    else:
        raise TypeError('input not recognized')

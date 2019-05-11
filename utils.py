import base64
import multiprocessing
import zlib


class CommonUtils(object):
    def __init__(self):
        # the number of byte to store index
        self.idx_byte = 2

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


class Code2File(CommonUtils):
    def __init__(self):
        super().__init__()

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

    def get_data(self, data, max_chunk):
        # decode letters to bytes
        data = self.string_to_byte_array(data)
        idx = self.byte_to_index(data[:self.idx_byte])
        if idx == 0:
            data_effective = data[2 * self.idx_byte:]
            max_chunk.value = self.byte_to_index(data[self.idx_byte:2 * self.idx_byte])
        else:
            data_effective = data[self.idx_byte:]
        return data_effective, idx


class File2Code(CommonUtils):
    def __init__(self, ops):
        super().__init__()
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
        bucket = [0, 0, 0, 0]
        chunks = []
        while i < len(byte_array):
            # check chunk length not to exceed 65536 files
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

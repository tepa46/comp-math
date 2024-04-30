import struct
from dataclasses import dataclass
import numpy as np

FLOAT_SIZE = 4
INT_SIZE = 4


@dataclass
class SVDOutput:
    def __init__(self, u: np.ndarray, s: np.ndarray, vh: np.ndarray):
        self.u = u
        self.s = s
        self.vh = vh

    def to_bytes(self):
        return self.u.astype(np.float32).tobytes() + \
               self.s.astype(np.float32).tobytes() + \
               self.vh.astype(np.float32).tobytes()

    @staticmethod
    def to_svd_output(bytes_array: bytes, height: int, width: int, k: int):
        float_array = [struct.unpack('<f', bytes_array[i: i + FLOAT_SIZE])
                       for i in range(0, len(bytes_array), FLOAT_SIZE)]
        float_np_array = np.array(float_array)

        u = float_np_array[: height * k].reshape(height, k)
        s = float_np_array[height * k: (height + 1) * k].ravel()
        vh = float_np_array[(height + 1) * k:].reshape(k, width)

        return SVDOutput(u, s, vh)

    def to_matrix(self) -> np.array:
        return self.u @ np.diag(self.s) @ self.vh


SVD_BMP_MAGIC = b'SVD-BMP'
SVD_BMP_MAGIC_SIZE = 7
SVD_METADATA_SIZE = SVD_BMP_MAGIC_SIZE + 3 * INT_SIZE  # magic, height, width, k


@dataclass
class SVDBmp:
    def __init__(self, height: int, width: int, k: int, red: SVDOutput, green: SVDOutput, blue: SVDOutput):
        self.height = height
        self.width = width
        self.k = k
        self.red = red
        self.green = green
        self.blue = blue

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            f.write(SVD_BMP_MAGIC)
            f.write(struct.pack('<I', self.height))
            f.write(struct.pack('<I', self.width))
            f.write(struct.pack('<I', self.k))
            f.write(self.red.to_bytes())
            f.write(self.green.to_bytes())
            f.write(self.blue.to_bytes())

    @staticmethod
    def open(file_path):
        with open(file_path, 'rb') as f:
            if f.read(SVD_BMP_MAGIC_SIZE) != SVD_BMP_MAGIC:
                raise TypeError(f"Incorrect file format: {file_path}")

            height = struct.unpack('<I', f.read(4))[0]
            width = struct.unpack('<I', f.read(4))[0]
            k = struct.unpack('<I', f.read(4))[0]

            red_svd = SVDOutput.to_svd_output(f.read((height + width + 1) * k * 4), height, width, k)
            green_svd = SVDOutput.to_svd_output(f.read((height + width + 1) * k * 4), height, width, k)
            blue_svd = SVDOutput.to_svd_output(f.read((height + width + 1) * k * 4), height, width, k)

        return SVDBmp(height, width, k, red_svd, green_svd, blue_svd)

import math
import os
from abc import ABC, abstractmethod
from svd_bmp import SVDBmp, SVDOutput, SVD_METADATA_SIZE
import numpy as np
from PIL import Image

BMP_CHANNEL_COUNT = 3
FLOAT_SIZE = 4


class SVDCompressor(ABC):

    @abstractmethod
    def _compress_channel(self, channel: np.ndarray, k: int) -> SVDOutput:
        ...

    def _compress(self, img_array: np.ndarray, k: int) -> tuple[SVDOutput, ...]:
        return tuple(self._compress_channel(img_array[..., i], k) for i in range(BMP_CHANNEL_COUNT))

    def to_svd(self, file_path: str, ratio: float) -> SVDBmp:
        # Image size with metadata
        img_size = os.path.getsize(file_path)

        img = Image.open(file_path)
        height = img.height
        width = img.width
        k = math.floor(img_size / (ratio * (SVD_METADATA_SIZE + FLOAT_SIZE * (height + width + 1))))

        img_arrays = np.asarray(img)
        r, g, b = self._compress(img_arrays, k)

        return SVDBmp(height, width, k, r, g, b)

    @staticmethod
    def from_svd(img: SVDBmp) -> Image:
        unpacked_arrays = [img.red.to_matrix(), img.green.to_matrix(), img.blue.to_matrix()]
        image_matrix = np.dstack(unpacked_arrays).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_matrix)


class SimpleSVDCompressor(SVDCompressor, ABC):
    def _compress_channel(self, channel: np.ndarray, k: int):
        u, s, vh = np.linalg.svd(channel, full_matrices=False)
        return SVDOutput(u[:, :k], s[:k], vh[:k, :])


class SVDPowerCompressor(SVDCompressor, ABC):

    def _power_svd(self, channel: np.ndarray, iterations: int):
        mu, sigma = 0, 1
        x = np.random.normal(mu, sigma, size=channel.shape[1])
        b = channel.T.dot(channel)
        for i in range(iterations):
            new_x = b.dot(x)
            x = new_x
        v = x / np.linalg.norm(x)
        sigma = np.linalg.norm(channel.dot(v))
        u = channel.dot(v) / sigma
        return np.reshape(u, (channel.shape[0], 1)), sigma, np.reshape(v, (channel.shape[1], 1))

    # Compute SVD using Power Method.
    # Reference link: http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf
    #
    # Auxiliary implementation: https://gist.github.com/Zhenye-Na/cbf4e534b44ef94fdbad663ef56dd333
    def _compress_channel(self, channel: np.ndarray, k: int):
        rank = np.linalg.matrix_rank(channel)
        ut = np.zeros((channel.shape[0], 1))
        st = []
        vht = np.zeros((channel.shape[1], 1))

        # Define the number of iterations
        delta = 0.001
        eps = 0.97
        lamda = 2
        iterations = int(math.log(
            4 * math.log(2 * channel.shape[1] / delta) / (eps * delta)) / (2 * lamda))

        # SVD using Power Method
        for i in range(rank):
            u, sigma, v = self._power_svd(channel, iterations)
            ut = np.hstack((ut, u))
            st.append(sigma)
            vht = np.hstack((vht, v))
            channel = channel - u.dot(v.T).dot(sigma)

        ut = ut[:, 1:]
        vht = vht[:, 1:]
        return SVDOutput(ut[:, :k], np.array(st)[:k], vht.T[:k, :])


class BlockSVDPowerCompressor(SVDCompressor, ABC):
    # https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html#j_jisys-2018-0034_fig_004
    def _compress_channel(self, channel: np.ndarray, k: int):
        eps = 10
        err = eps + 0.1

        u = np.zeros((channel.shape[0], k))
        s = np.zeros(k)
        vh = np.zeros((channel.shape[1], k))

        while err > eps:
            q, _ = np.linalg.qr(np.dot(channel, vh))
            u = q[:, :k]

            q, r = np.linalg.qr(np.dot(channel.T, u))
            vh = q[:, :k]
            s = r[:k, :k]

            err = np.linalg.norm(np.dot(channel, vh) - np.dot(u, s))

        return SVDOutput(u, np.diag(s).astype(np.float32), vh.T)

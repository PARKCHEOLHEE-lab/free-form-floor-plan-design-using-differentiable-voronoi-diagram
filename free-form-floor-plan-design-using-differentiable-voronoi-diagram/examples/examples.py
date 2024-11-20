import numpy as np
import matplotlib.pyplot as plt

from shapely import geometry
from svgpathtools import parse_path


class SvgPathParser:
    def __init__(self, path, normalize):
        self.path = path
        self.normalize = normalize

        self._parse()

    def _parse(self):
        self.svg_path = parse_path(self.path)

        coordinates = []
        for segment in self.svg_path:
            start = (segment.start.real, segment.start.imag)
            coordinates.append(start)

        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])

        self.coordinates = np.array(coordinates)

        if self.normalize:
            self.coordinates -= self.coordinates.mean(axis=0)
            self.coordinates /= np.linalg.norm(self.coordinates, axis=1).max()

        self.boundary_polygon = geometry.Polygon(self.coordinates)

    def show(self):
        x, y = zip(*self.coordinates)
        plt.plot(x, y, marker="o")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


class Duck(SvgPathParser):
    def __init__(self, normalize=True):
        duck_path = """M7920 11494 c-193 -21 -251 -29 -355 -50 -540 -105 -1036 -366 -1442
            -758 -515 -495 -834 -1162 -904 -1891 -15 -154 -6 -563 15 -705 66 -440 220
            -857 442 -1203 24 -37 44 -69 44 -71 0 -2 -147 -3 -327 -4 -414 -1 -765 -23
            -1172 -72 -97 -12 -167 -17 -170 -11 -3 5 -33 52 -66 106 -231 372 -633 798
            -1040 1101 -309 229 -625 409 -936 532 -287 113 -392 130 -500 79 -65 -32
            -118 -81 -249 -237 -627 -745 -1009 -1563 -1170 -2505 -54 -320 -77 -574 -86
            -965 -28 -1207 238 -2308 785 -3242 120 -204 228 -364 270 -397 84 -67 585
            -319 901 -454 1197 -511 2535 -769 3865 -744 983 19 1875 166 2783 458 334
            108 918 340 1013 404 99 65 407 488 599 824 620 1080 835 2329 614 3561 -75
            415 -226 892 -401 1262 -39 82 -54 124 -47 133 5 7 42 58 82 114 41 55 77 99
            81 96 4 -2 68 -8 142 -14 766 -53 1474 347 1858 1051 105 192 186 439 228 693
            27 167 24 487 -6 660 -33 189 -64 249 -150 289 -46 21 -51 21 -846 21 -440 0
            -828 -3 -861 -7 l-62 -7 -32 86 c-54 143 -194 412 -289 554 -479 720 -1201
            1178 -2040 1295 -101 14 -496 27 -571 18z
        """

        super().__init__(duck_path, normalize)


class ShapePathParser:
    def __init__(self, path, normalize):
        self.path = path
        self.normalize = normalize

        self._parse()

    def _parse(self):
        self.coordinates = np.array(self.path)

        if self.normalize:
            self.coordinates -= self.coordinates.mean(axis=0)
            self.coordinates /= np.linalg.norm(self.coordinates, axis=1).max()

        self.boundary_polygon = geometry.Polygon(self.coordinates)

    def show(self):
        x, y = zip(*self.coordinates)
        plt.plot(x, y, marker="o")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


class ShapeA(ShapePathParser):
    def __init__(self, normalize=True):
        path = [
            [-51.2, -26.2],
            [276.8, -26.2],
            [276.8, 94.2],
            [218.0, 94.2],
            [218.0, 188.5],
            [197.8, 188.5],
            [197.8, 254.4],
            [100.8, 254.4],
            [100.8, 234.8],
            [0.0, 234.8],
            [0.0, 65.9],
            [-51.2, 65.9],
            [-51.2, -26.2],
        ]

        super().__init__(path, normalize)


class ShapeB(ShapePathParser):
    def __init__(self, normalize=True):
        path = [
            [36.1, 42.4],
            [109.9, 42.4],
            [109.9, 57.4],
            [136.3, 57.4],
            [136.3, 76.2],
            [183.8, 76.2],
            [183.8, 139.2],
            [210.2, 139.2],
            [210.2, 191.5],
            [89.5, 191.5],
            [89.5, 178.9],
            [36.1, 178.9],
            [36.1, 43.8],
        ]

        super().__init__(path, normalize)


class ShapeC(ShapePathParser):
    def __init__(self, normalize=True):
        path = [
            [48.5, 59.4],
            [88.5, 59.4],
            [88.5, 88.0],
            [126.6, 88.0],
            [126.6, 59.4],
            [178.4, 59.4],
            [178.4, 113.4],
            [197.7, 113.4],
            [197.7, 172.5],
            [48.5, 172.5],
            [48.5, 59.4],
        ]

        super().__init__(path, normalize)

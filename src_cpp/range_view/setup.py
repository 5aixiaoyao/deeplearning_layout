from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name = 'cudaDemo',
    packages=find_packages(),
    version='0.1.0',
    author="LCW",
    ext_modules=[
        CUDAExtension('range_view',
        ['./ops/src/range_view/range_view.cpp',
        './ops/src/range_view/range_view_cuda.cu',]
        ),
        CUDAExtension('grid2points',
        ['./ops/src/grid2points/grid2points.cpp',
        './ops/src/grid2points/grid2points_cuda.cu',]
        ),
        CUDAExtension('points2grid',
        ['./ops/src/points2grid/points2grid.cpp',
        './ops/src/points2grid/points2grid_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


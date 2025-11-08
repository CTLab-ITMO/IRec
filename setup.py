from setuptools import setup, find_packages
from pathlib import Path

install_requires = [
    'numpy',
    'lmdb',
    'loguru',
    'murmurhash',
    'scikit-learn',
    'tensorboard',
    'tqdm',
    'onnx',
    'brotli',
    'lz4',
    'mpi4py',
    'pynvml',
    'zstandard',
]

with open(Path(__file__).parent / 'README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='irec',
    version='1.0.0',
    package_dir={'': 'src'},
    description='Framework for R&D RecSys projects',
    author='Vladimir Baikalov',
    author_email='nonameuntitled159@gmail.com',
    url='https://github.com/CTLab-ITMO/IRec',
    packages=find_packages(
        where='src',
        exclude=['tests', 'tests.*', '*.tests', '*.tests.*']
    ),
    install_requires=install_requires,
    python_requires='>=3.12',
    
)
import codecs
import os.path
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1].strip()
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='seg_training',
    version=get_version('seg_training/__init__.py'),
    description='TASFormer',
    long_description='TASFormer',
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='',
    packages=[
        'seg_training',
        'seg_training.networks',
        'seg_training.utils',
        'seg_training.datasets',
    ],
    license="MIT",
    install_requires=[
        'nnio>=0.3.0',
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
        'albumentations',
        'pytorch-lightning',
        'einops',
        'wandb',
    ]
)

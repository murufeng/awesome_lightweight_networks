from setuptools import setup, find_packages

setup(
  name = 'light_cnns',
  packages = find_packages(),
  version = '0.4.0',
  license='MIT',
  description = 'Implementation of Lightweight Network in Pytorch',
  author = 'murufeng',
  author_email = '2923204420@qq.com',
  url = 'https://github.com/murufeng/awesome_lightweight_networks',
  keywords = [
    'artificial intelligence',
    'lightweight neural network',
    'image classification',
    'image segmentation',
    'computer vision'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.4',
    'torchvision'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
  ],
)

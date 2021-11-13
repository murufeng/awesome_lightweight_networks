from setuptools import setup, find_packages

setup(
  name = 'lightnet',
  packages = find_packages(exclude=['examples']),
  version = '0.1.0',
  license='MIT',
  description = 'Implementation of Lightweight Network in Pytorch',
  author = 'murufeng',
  author_email = '2923204420@qq.com',
  url = 'https://github.com/murufeng/awesome_lightweight_networks',
  keywords = [
    'artificial intelligence',
    'lightweight neural network',
    'image classification',
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
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
  ],
)

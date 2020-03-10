from setuptools import setup, find_packages
 
setup(name='obstacle_avoidance',
      version='0.1',
      url='',
      license='MIT',
      author='Shamsi Abdullayev',
      author_email='shamsi@gabdullayev.net',
      description='Source code for training a small robot car to avoid obstacles',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)
from setuptools import setup, find_packages
from distutils.util import convert_path


ver_path = convert_path('cw/version.py')
with open(ver_path) as ver_file:
    ns = {}
    exec(ver_file.read(), ns)
    version = ns['version']

setup(
    name='cw',
    version=ns['version'],
    description="Collection of python libraries I've written throughout the years.",
    author='Aaron de Windt',
    author_email='',
    url='https://github.com/aarondewindt/cw',

    install_requires=['numpy',
                      'scipy',
                      'msgpack',
                      'PyYAML',
                      'python-dateutil',
                      'cerberus',
                      'pyserial',
                      'tqdm',
                      "openpyxl",
                      "psutil",
                      "pyproj",
                      "requests",
                      "matplotlib",
                      "xarray",
                      "sympy",
                      "control",
                      "msgpack_numpy",
                      "pandas"],
    packages=find_packages('.', exclude=["test"]),
    package_data={
        "cw.aero_file": "*.yaml",
    },
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha'],
    entry_points={
        'console_scripts': [
            'cw = cw.__main__:main'
        ]
    }
)

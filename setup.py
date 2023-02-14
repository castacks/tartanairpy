from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'CI-TartanAir'
LONG_DESCRIPTION = 'Testing TartanAir'

setup(
    name="convrsn",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=" ",
    author_email="kunalkapoor@cmu.edu",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='conversion',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)

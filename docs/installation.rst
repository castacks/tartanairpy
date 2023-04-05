

Installation
=====================================
To install the TartanAir Python package:

.. code-block:: bash

    pip install tartanair

TartanAir depends on PyTorch for some customization functionalities. Please install it on your system (we don't do it for you to avoid version conflicts). Please follow the instructions on the PyTorch website: https://pytorch.org/get-started/locally/ .
That's it! You're ready to go.

**Note while we are in development, please install using the following command:**   

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ tartanair

Requirements
------------

Currently the TartanAir Python package was only tested on Ubuntu 20.04. We will be testing on other operating systems in the future.

Known Installation Issues and Solutions
---------------------------------------

Ubuntu
~~~~~~
1. Downloading does not work. It could be that you are missing `wget`. Get it using the following command:

    .. code-block:: bash

        sudo apt install wget

MacOS
~~~~~
1. `PyYAML` fails to install with `pip` on MacOS. Please install it manually using the following command:

    .. code-block:: bash

        python3 -m pip install pyyaml

2. `opencv-contrib-python` fails to install with `pip` on MacOS. Please install it manually using the following command:

    .. code-block:: bash

        python3 -m pip install opencv-contrib-python

    This might take a while.

3. `pytransform3d` fails to install with `pip` on MacOS. Please install it manually using the following command:

    .. code-block:: bash

        python3 -m pip install pytransform3d

4. `Pillow` can cause trouble with older versions of `torchvision`. If you are facing issues with `Pillow`, like `ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'`, please install it manually using the following command, remove the `torchvision` package and install it again:

    .. code-block:: bash

        python3 -m pip install Pillow
        python3 -m pip uninstall torchvision
        python3 -m pip install torchvision

5. `wget` is not installed by default on MacOS. Please install it manually using the following command:

    .. code-block:: bash

        brew install wget

6. URLLib may not find your certificates on Mac, and you'll see something like

    .. code-block:: bash

        urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1123)>

    Please run this in terminal to fix (adapt to your Python version):
    
        .. code-block:: bash
    
            /Applications/Python\ 3.8/Install\ Certificates.command


Installation
=====================================
To install the TartanAir Python package:

.. code-block:: bash

    pip install tartanair

That's it! You're ready to go.

**Note while we are in development, please install using the following command:**   

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ tartanair

Requirements
------------

Currently the TartanAir Python package was only tested on Ubuntu 20.04. We will be testing on other operating systems in the future.

Known Installation Issues and Solutions
---------------------------------------

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

5. `wget` is not installed properly by `pip`. Please install it manually using the following command:

    .. code-block:: bash

        python3 -m pip install wget
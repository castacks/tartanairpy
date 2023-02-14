# TartanAir

Hello and welcome to the official TartanAir repository. This repository includes a set of tools that complement the [TartanAir Dataset](https://www.tartanair.org/)). 

# Dear developers please read this.
To be able to run our tests and the package overall, while we are still developing it, please create a `.txt` file somewhere with an Azure access token, and point `src/tartanair/taratanair_module.py` to it. Look for the place to do that at the bottom of the file. This is a temporary solution and will be changed as soon as we move the data to a public blob.


## Installation
```
pip install tartanair
```

## Usage
```
from tartanair import TartanAir
ta = TartanAir()
```

## Examples
You may run the examples provided in the `examples` folder. One option for trying those out directly is to clone the repository and run the following command from the examples directory:
```
$ cd examples
$ python3 <example_name>.py
```

**Notice: The examples, when run directly from a cloned repo, require the `tartanair` package to be uninstalled if already installed. This may be changed soon but for now please only have one copy of the package.**


## Citation
If you use this dataset, please cite the following paper:
<!-- TODO(yoraish) -->
```
@inproceedings{wang2023pip,
  title={pip install tartanair: },
  author={},
  booktitle={},
  year={2023}
}
```

## License
MIT License. 

## Contributing.
The structure of the repository is as follows:
```
src
├── tartanair
│   ├── tartanair_module.py # Super class for all modules. Each module is instantiated in the TartanAir class.
│   ├── downloader.py
│   ├── visualizer.py
│   ├── customizer.py
│   ├── ...
│   └── __init__.py

tests
├── test_downloader.py
├── test_visualizer.py
├── test_customizer.py
└── ...

examples
├── example_downloader.py
├── example_visualizer.py
├── example_customizer.py
└── ...
```

Please add contribution of new modules as follows. Let's use an example of a new module for underwatering the dataset. This would implement the method `tartanair.underwaterify()` and would make everything in the dataset look like it was taken underwater. Do we want to do this? Yes. But let's pretend we don't.

1. Create a new module in `src/tartanair`, in our case `underwaterifyer.py`.
2. In `src/tartanair/underwaterifyer.py`, create a class `TartanAirUnderwaterifyer` that inherits from `TartanAirModule`. This class should implement the method `underwaterify()`.
3. In `src/tartanair/tartanair.py`, instantiate the class `TartanAirUnderwaterifyer` in the `__init__()` method of the `TartanAir` class and add the method `underwaterify()` that calls the method `underwaterify()` of the `TartanAirUnderwaterifyer` class.
4. Add tests for the new module in `tests/test_underwaterifyer.py`.
5. Add an example for the new module in `examples/example_underwaterifyer.py`.




# TartanAir-V2 Python Package Sanity Tests.

Hello and welcome to datasets with friends. May 2023 limited edition. Sit back, relax, and keep an eye out for bugs.

This directory hosts a set of _very simple and absolutely not comprehensive tests_. The tests serve as a unified example for using and testing various functionalities of the TartanAir-V2 Python package.

## How Do I Run the Tests?
Great question. You can run the tests by running the following command from the root of the repository:
```bash
python3 tartanairpy_test.py 
```

**Note**: for any test the requires data download (many of the tests download a small amount of data), you must set the `self.azure_token` member variable.

### Running Specific Tests

```bash
python3 tartanairpy_test.py TartanAirTest.test_customization
```

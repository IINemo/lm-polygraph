This set of tests is intended to prevent regressions in the benchmark. It tests for all the benchmarks to run with 0 exit code, save the manager correctly and be run against the correct model inputs.

These tests are supposed to be run in the environment with the available CUDA device, otherwise this suite will take a long time to run and might be flaky.

`fixtures` directory contains expected model inputs with correct prompting for all combinations of methods and datasets.

As provided this suite contains pre-generated fixtures that reflect the state of the datasets and prompting which was used at the time of TACL submission. If at any point the datasets or prompting change, the fixtures will need to be regenerated. This can be done by running the `generate_fixtures.py` script in the `fixtures` directory. This script will generate new fixtures for all combinations of methods and datasets, and mark a new state to be maintained by the tests.

This set of tests is intended to prevent regressions in the benchmark. It tests for all the benchmarks to run with 0 exit code, save the manager correctly and be run against the correct model inputs.

These tests are supposed to be run in the environment with the available CUDA device, otherwise this suite will take a long time to run and might be flaky.

`fixtures` directory contains expected model inputs with correct prompting for all combinations of methods and datasets.

As provided this suite contains pre-generated fixtures that reflect the state of the datasets and prompting which was used at the time of TACL submission. If at any point the datasets or prompting change, the fixtures will need to be regenerated. This can be done by running the `generate_fixtures.py` script in the `fixtures` directory. This script will generate new fixtures for all combinations of methods and datasets, and mark a new state to be maintained by the tests.

!!!IMPORTANT!!!: This suite does not cover density-based methods, since covariance matrix is ill-conditioned for such a small number of samples. Thus, regressions for these methods can occur without turning these tests red. It also does not check for claim-level experiments to return 0, since they require OpenAI API key to run, tests only check for correct model inputs.

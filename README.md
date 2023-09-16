# Privacy-preserving in-context learning with differentially private few-shot generation
This is a codebase to perform privacy-preserving in-context learning with differentially private few-shot generation.

## Experiments
See `run.sh` for example commands for AGNEWS/DBPedia/TREC/MIT-G/MIT-D. Here are a few explanations for the parameters in `run.sh`.

```
--sigma 0.39                  # noise parameter
--openai-model "babbage"      # openai model
--num-private-train 20        # num_private_train=MN. MN=0 and M=0 with num_valid=4 will get epsilon=0 (4-shot) results.
--set-num-public-train 0      # by default set to 0. set_num_public_train >0 indicates additonal public data available.
--num-valid 4                 # num-valid=n. n samples to be generated for n-shot ICL
--num-private-train-splits 10 # num_private_train_splits=M
--num-test 1000               # if len(test_set)<1000, use the exact test set. Otherwise we sample 1000 samples from test set for evaluation
--use-dp-prompts              # generate prompts from private dataset
--sample-same-label-prompts   # sample_same_label_prompts=True, sample subsets from the sets with same targeted labels.
--subsample-per-token         # subsample_per_token=True, at each token generation, subsample a fresh new subset.
--no-public-token             # no_public_token=True, RVP=False
--synth-seed 0                # random seed for subsampling in generation
--eval-seed 0                 # random seed for n-shot demonstrations sampling in evaluation
```

Note: Due to the randomness in generations caused by DP noise, the results may be slightly different from the reported values in the paper.

### About reproducing experiments with OpenAI models
Our code uses the `logprobs` parameter of OpenAI's API (https://platform.openai.com/docs/api-reference/completions/create#logprobs) with a value of 100.
By default, OpenAI currently allows up to 5 as the value for `logprobs`. Unless you obtain permission from OpenAI to use a larger value, the code will not work as-is.
The existing code uses models which have been [deprecated by OpenAI](https://platform.openai.com/docs/deprecations/base-gpt-models) and may no longer be available in the future.

As an alternative, you can consider using alternative LMs through software like https://github.com/vllm-project/vllm which provides an OpenAI-compatible API.
It's also possible to use the `logit_bias` parameter (https://platform.openai.com/docs/api-reference/completions/create#logit_bias) to get top-k log probs for larger values of k
by repeatedly querying the API with the same prefix while banning the most likely tokens obtained so far.


## Setup
1. Install Python 3.10.

   One way is to use [pyenv](https://github.com/pyenv/pyenv).
   Run `pyenv install --list | grep '^ *3.10' | tail -n1` to discover the most recent minor version of Python 3.10.
   Run `pyenv install 3.10.X` where `X` is the latest minor version available.

1. Install [Poetry](https://python-poetry.org/) following the [instructions](https://python-poetry.org/docs/#installation).
1. Configure Poetry to use your Python 3.10 installation.
    - If using `pyenv` setup above: run `poetry env use $(pyenv prefix 3.10.X)/bin/python`
    - Otherwise: run `poetry env use <path to your Python 3.10 binary>`
1. Run `poetry install` to install the dependencies.

## IDE Setup
### IntelliJ/PyCharm
- IntelliJ only: Install the Python plug-in.
- Setup the Python Interpreter
    - PyCharm: open Settings then go to Python Interpreter.
    - IntelliJ: go to `File -> Project Structure -> Project -> Project SDK`.
    - Ensure that the Python environment in the `.venv` directory is selected. If needed, you can add a new interpreter.
      Choose "Poetry Environment" as the interpreter type. Select to use "Existing environment" with the interpreter in `.venv/bin/python`.
- Setup source folders
    - Right click the `src` folder and choose `Mark Directory As -> Sources Root`.
    - Right click the `tests` folder and choose `Mark Directory As -> Test Sources Root`.
- Configure `pytest`: In `Preferences -> Tools -> Python Integrated Tools`, set the default test runner to `pytest`.
- ruff: You can try a [plugin](https://plugins.jetbrains.com/plugin/20574-ruff) as a replacement for running `make lint` manually.

### Visual Studio Code
- Install the Python extension.
- Open the root directory of the project.
- Open the Command Palette and choose "Python: Select Interpreter". Ensure the one in `.venv` is selected.
  If not, you can choose "Enter interpreter path..." and enter `./.venv/bin/python`.
- Configure `pytest`: open the Command Palette and choose "Python: Configure Tests". Choose pytest. Chooses `tests` as the root directory for tests.
- ruff: You can try an [extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) if you want.

## Development
We have automated code style checks, linting for common errors, type checking, and unit testing.
- Run `make` to run formatting, linting, type checking, and unit testing, in that order.
  You can run each of the four checks separately with `make format-check`, `make lint`, `make pyright`, and `make pytest`.
- Run `make format` and `make lint-fix` to automatically fix formatting errors and (some) linting errors.

## Project structure
- `src/`: Python code for the project.
- `tests/`: Unit tests for code in `src/`.
- `data/`: Python code for data processing of MIT dataset.
- `privacy_analysis/`: Python code for calculating the noise parameter.
- `lmapi/`: a custom wrapper for OpenAI's API.

## Acknowledgments
This project is built upon the foundation of [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://github.com/tonyzhaozh/few-shot-learning).
We would like to thank the contributors and maintainers of the original repository for their valuable work.

## Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

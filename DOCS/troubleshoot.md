## Troubleshooting Tips

- If you encounter trouble installing `africanwhisper` package on Kaggle, and encounter the error:
```commandline
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: '/opt/conda/lib/python3.10/site-packages/aiohttp-3.9.1.dist-info/METADATA'
```
Execute this command before installing the package:
```commandline
!rm /opt/conda/lib/python3.10/site-packages/aiohttp-3.9.1.dist-info -rdf
```
see [Issue #142](https://github.com/KevKibe/African-Whisper/issues/142) for more info.


- If you encounter this error installing `africanwhisper` package on Colab:
```commandline
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
spacy 3.7.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.12.3 which is incompatible.
torchtext 0.18.0 requires torch>=2.3.0, but you have torch 2.2.2 which is incompatible.
weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.12.3 which is incompatible.
Successfully installed GitPython-3.1.43 Mako-1.3.5 accelerate-0.29.3 africanwhisper-0.9.5 alembic-1.13.1 antlr4-python3-runtime-4.9.3 appdirs-1.4.4 asteroid-filterbanks-0.4.0 av-11.0.0 bitsandbytes-0.42.0 coloredlogs-15.0.1 colorlog-6.8.2 ctranslate2-4.1.0 datasets-2.17.0 dill-0.3.8 docker-pycreds-0.4.0 docopt-0.6.2 einops-0.8.0 evaluate-0.4.1 faster-whisper-1.0.0 gitdb-4.0.11 h5py-3.11.0 holoviews-1.18.3 humanfriendly-10.0 hyperpyyaml-1.2.2 jiwer-3.0.3 julius-0.2.7 keras-3.1.1 librosa-0.10.1 lightning-2.2.5 lightning-utilities-0.11.2 ml-dtypes-0.3.2 multiprocess-0.70.16 namex-0.0.8 numpy-1.26.4 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.5.40 nvidia-nvtx-cu12-12.1.105 omegaconf-2.3.0 onnxruntime-1.18.0 optree-0.11.0 optuna-3.6.1 peft-0.10.0 primePy-1.3 pyannote-audio-3.1.1 pyannote.core-5.0.0 pyannote.database-5.1.0 pyannote.metrics-3.2.1 pyannote.pipeline-3.0.1 python-dotenv-1.0.1 pytorch-lightning-2.2.5 pytorch-metric-learning-2.5.0 rapidfuzz-3.9.3 responses-0.18.0 ruamel.yaml-0.18.6 ruamel.yaml.clib-0.2.8 scipy-1.12.0 semver-3.0.2 sentry-sdk-2.4.0 setproctitle-1.3.3 shellingham-1.5.4 smmap-5.0.1 speechbrain-1.0.0 tensorboard-2.16.2 tensorboardX-2.6.2.2 tensorflow-2.16.1 tensorflow-probability-0.24.0 tf-keras-2.16.0 tokenizers-0.15.2 torch-2.2.2 torch-audiomentations-0.11.1 torch-pitch-shift-1.2.4 torchaudio-2.2.2 torchmetrics-1.4.0.post0 torchvision-0.17.2 transformers-4.39.2 triton-2.2.0 typer-0.12.3 wandb-0.16.6 xxhash-3.4.1
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
WARNING: The following packages were previously imported in this runtime:
  [pydevd_plugins]
You must restart the runtime in order to use newly installed versions.
```
restart the kernel and continue with the next step.

- If you encounter the error:
```commandline
TypeError: expected string or bytes-like object
```
upgrade `pandas` version to `2.2.2` and restart kernel 


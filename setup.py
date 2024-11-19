from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

BASE_DEPS = [
    "transformers==4.42.3",
    "datasets==2.19.2",
    "librosa==0.10.2.post1",
    "evaluate==0.4.1",
    "jiwer==3.0.4",
    "bitsandbytes==0.42.0",
    "accelerate==0.31.0",
    "peft==0.11.1",
    "numpy==1.26.4",
    "wandb==0.17.4",
    "holoviews==1.18.3",
    "panel==1.3.8",
    "tf-keras==2.16.0",
    "tensorflow==2.16.1",
    "keras==3.1.1",
    "scipy==1.12.0",
    "tensorflow-probability==0.24.0",
    "faster-whisper==1.0.3",
    "python-dotenv==1.0.1",
    "pyannote-audio==3.2.0",
    "nltk==3.8.1",
    "torchvision",
    "ctranslate2==4.4.0",
    "pandas==2.2.2",
    "huggingface_hub",
    "soundfile",
    "tqdm"
]

DEPLOYMENT_DEPS = [
    "torch",
    "transformers==4.42.3",
    "pydantic==2.7.3",
    "prometheus-client==0.20.0",
    "fastapi==0.111.0",
    "uvicorn==0.30.1",
    "python-dotenv==1.0.1",
    "faster-whisper==1.0.3",
    "pyannote-audio==3.2.0",
    "nltk==3.8.1",
    "torchvision",
    "ctranslate2==4.4.0",
    "pandas==2.2.2",
]
ALL_DEPS = BASE_DEPS + DEPLOYMENT_DEPS

setup(
    name="africanwhisper",
    version="0.9.20",
    author="Kevin Kibe",
    author_email="keviinkibe@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={
        "deployment.faster_whisper": ["mel_filters.npz"],
    },
    include_package_data=True,
    url="https://kevkibe.github.io/African-Whisper",
    project_urls={
        "Source": "https://github.com/KevKibe/African-Whisper",
    },
    description = "A framework for fast fine-tuning and API endpoint deployment of Whisper model specifically developed to accelerate Automatic Speech Recognition(ASR) for African Languages.",
    readme = "README.md",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "MIT",
    python_requires=">=3.9",
    # install_requires = BASE_DEPS,
    extras_require={
        "all": ALL_DEPS,
        "training": BASE_DEPS,
        "deployment": DEPLOYMENT_DEPS
    },
    classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    ]
)

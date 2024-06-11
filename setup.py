from setuptools import find_packages, setup

BASE_DEPS = [
    "transformers==4.39.2",
    "datasets==2.17.0",
    "librosa==0.10.1",
    "evaluate==0.4.1",
    "jiwer==3.0.3",
    "bitsandbytes==0.42.0",
    "accelerate==0.29.3",
    "peft==0.10.0",
    "numpy==1.26.4",
    "wandb==0.16.6",
    "holoviews==1.18.3",
    "panel==1.3.8",
    "tf-keras==2.16.0",
    "tensorflow==2.16.1",
    "keras==3.1.1",
    "scipy==1.12.0",
    "tensorflow-probability==0.24.0",
    "faster-whisper==1.0.0",
    "python-dotenv==1.0.1",
    "pyannote-audio==3.1.1",
    "nltk==3.8.1",
    "torchvision==0.17.2",
    "ctranslate2==4.1.0",
    "pandas==2.0.3",
]

DEPLOYMENT_DEPS = [
    "torch==2.2.2",
    "transformers==4.39.1",
    "pydantic==2.6.4",
    "prometheus-client==0.20.0",
    "fastapi==0.110.1",
    "uvicorn==0.29.0",
    "python-dotenv==1.0.1",
    "faster-whisper==1.0.0",
    "pyannote-audio==3.1.1",
    "nltk==3.8.1",
    "torchvision==0.17.2",
    "ctranslate2==4.1.0",
    "pandas==2.2.1",
]
ALL_DEPS = BASE_DEPS + DEPLOYMENT_DEPS

setup(
    name="africanwhisper",
    version="0.9.6",
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
    python_requires=">=3.9",
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

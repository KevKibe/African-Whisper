from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='africanwhisper',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    author='Kevin Kibe',
    author_email='keviinkibe@gmail.com',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KevKibe/African-Whisper',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

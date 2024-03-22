from setuptools import setup, find_packages



setup(
    name='africanwhisper',
    version='0.2-beta',
    packages=find_packages(),
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

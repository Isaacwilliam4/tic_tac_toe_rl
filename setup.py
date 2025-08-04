from setuptools import setup, find_packages

setup(
    name='tic_tac_rl',
    version='0.1.0',
    author='Isaac Peterson',
    description='Reinforcement learning experiments on Tic Tac Toe from Button and Sutton (2020)',
    packages=find_packages(),
    install_requires=['numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.8',
)

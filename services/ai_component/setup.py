from setuptools import setup, find_packages

setup(
    name='ai_component',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'mujoco-py',
        'torch',
        'numpy',
        'pyyaml',
        'matplotlib',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A configurable reinforcement learning library implementing PPO, SAC, DDPG, and more.',
    url='https://github.com/yourusername/ai_component',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

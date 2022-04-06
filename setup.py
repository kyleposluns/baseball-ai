from setuptools import setup

setup(
    name="baseball-ai",
    version="1.0",
    packages=["baseball_ai", "baseball_ai.scripts"],
    install_requires=["marshmallow==3.13.0", "jsonstream==0.0.1", "rstr==3.1.0"],
    author="kyle",
    entry_points={
        "console_scripts": [
            "gen-training-data=baseball_ai.scripts.generate_training_data:main",
        ]
    },
)

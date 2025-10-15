from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="agent",
    version="0.1.0",
    author="hse",
    description="AI4AI agent implementation",
    packages=find_packages(),
    package_data={
        "agent": [
            "../requirements.txt",
            "utils/config.yaml",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agent = agent.run:run",
        ],
    },
)
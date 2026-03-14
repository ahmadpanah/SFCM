from setuptools import setup, find_packages
setup(
    name="sfcm", version="1.0.0",
    description="Surrogate-Feedback Co-evolutionary Metaheuristic",
    author="Seyed Hossein Ahmadpanah",
    author_email="h.ahmadpanah@iau.ac.ir",
    url="https://github.com/ahmadpanah/sfcm",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*","docker*"]),
    install_requires=[
        "torch>=2.0.0","numpy>=1.24.0","scipy>=1.10.0",
        "scikit-learn>=1.2.0","pandas>=2.0.0","matplotlib>=3.7.0","pyyaml>=6.0",
    ],
    extras_require={
        "baselines": ["smac>=2.0.0","cma>=3.3.0","gpytorch>=1.10"],
        "dev":       ["pytest>=8.0","pytest-cov>=4.0"],
    },
)

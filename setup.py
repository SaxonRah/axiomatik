from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="axiomatik",
    version="0.2.0",
    author="SaxonRah",
    author_email="paraboliclabs@gmail.com",  # Replace with your email
    description="Axiomatik is a comprehensive runtime verification system that brings formal verification concepts to practical Python programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaxonRah/axiomatik",
    project_urls={
        "Bug Tracker": "https://github.com/SaxonRah/axiomatik/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Assumes MIT. Adjust if needed.
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "click==8.2.1",
        "libcst==1.8.2",
        "matplotlib==3.10.5"
    ],
)

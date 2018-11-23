import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="priority_memory",
    version="0.0.2",
    author="Zixuan Chen",
    author_email="remch183@outlook.com",
    description="A prioritized sampling tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rem2016/priority_memory_buffer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
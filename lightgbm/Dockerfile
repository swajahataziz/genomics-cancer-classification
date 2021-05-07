FROM ubuntu:18.04

MAINTAINER Wajahat Aziz <syedazi@amazon.com>

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Installing Python3
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
        tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
        ./configure && make && make install && \
        apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
        make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
        ln -s /usr/local/bin/pip3 /usr/bin/pip

# Upgrading pip and creating symbolic link for python3
RUN ${PIP} --no-cache-dir install --upgrade pip
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN ${PIP} install \
        numpy==1.16.2 \
        pandas \
        scikit-learn==0.22 \
        requests==2.21.0 \
        scipy==1.2.1 \
        imbalanced-learn \
        lightgbm \
        pyHSICLasso \
        tqdm

RUN apt-get update -y
RUN apt-get install -y libgomp1

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache --upgrade \
    sagemaker-training

# Copies code under /opt/ml/code where sagemaker-containers expects to find the script to run
COPY program/* /opt/ml/code/

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py
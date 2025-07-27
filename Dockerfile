FROM ubuntu:22.04


ENV DEBIAN_frontend noninteractive
ENV TZ=Etc/UTC
ENV LANG en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y \
    sudo \
    git \
    unzip \
    tar \
    nano \
    vim \
    gedit \
    make \
    cmake \
    curl \
    python3-pip \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --gecos '' captain \
    && adduser captain sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER captain
WORKDIR /home/captain/
RUN chmod a+rwx /home/captain/


RUN sudo apt-get update && sudo apt install -y software-properties-common && \
    sudo add-apt-repository universe && \
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    sudo apt install ros-noetic-ros-base && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    sudo rm -rf /var/lib/apt/lists/*

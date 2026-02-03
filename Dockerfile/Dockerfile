FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04


ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_frontend=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=C.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


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


RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    sudo apt update && sudo apt install -y ros-noetic-desktop-full && \
    # source /opt/ros/noetic/setup.bash && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    sudo rosdep init && rosdep update && \
    sudo rm -rf /var/lib/apt/lists/*

COPY train/requirements.txt .

RUN pip3 install -r "requirements.txt"

RUN rm requirements.txt

RUN sudo apt update && \
sudo apt upgrade -y && \
sudo apt-get install ros-noetic-realsense2-camera -y && \
sudo apt install ros-noetic-foxglove-bridge -y


# # Загрузка и установка Miniconda
# RUN /bin/bash /home/captain/visual-transformer/Miniconda3.sh -b -p $CONDA_DIR && \
#     $CONDA_DIR/bin/conda clean -tipsy
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=captain

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


RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME} && \
    usermod -a -G sudo,dialout ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${USERNAME}
WORKDIR /home/$USERNAME/
RUN sudo chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/


RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    sudo apt update && sudo apt install -y ros-noetic-desktop-full && \
    # source /opt/ros/noetic/setup.bash && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    sudo rosdep init && rosdep update && \
    sudo rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r "requirements.txt"

RUN rm requirements.txt

RUN sudo apt update && \
    sudo apt upgrade -y && \
    # sudo apt-get install ros-noetic-realsense2-camera -y && \
    sudo apt install ros-noetic-foxglove-bridge -y && \
    # USB support for Orbbec Femto Bolt camera
    sudo apt install -y libusb-1.0-0-dev libudev-dev usbutils && \
    sudo rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/noetic/setup.bash && sudo apt-get update && \
    sudo apt install -y libgflags-dev ros-$ROS_DISTRO-image-geometry ros-$ROS_DISTRO-camera-info-manager \
        ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-publisher libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev \
        ros-$ROS_DISTRO-diagnostic-updater ros-$ROS_DISTRO-diagnostic-msgs libdw-dev && \
    mkdir -p "thirdparty_ws/src" && cd thirdparty_ws/src && \
    git clone https://github.com/orbbec/OrbbecSDK_ROS1.git && \
    catkin_init_workspace && \
    cd .. && catkin_make && \
    sudo rm -rf /var/lib/apt/lists/*

RUN source /opt/ros/noetic/setup.bash && sudo apt-get update && \
    sudo apt install -y ros-$ROS_DISTRO-teleop-twist-keyboard \
        ros-$ROS_DISTRO-joint-state-publisher-gui \
        ros-$ROS_DISTRO-ros-controllers \
        libasio-dev && \
    cd "thirdparty_ws/src" && \
    git clone https://github.com/agilexrobotics/ugv_sdk.git && \
    git clone https://github.com/agilexrobotics/tracer_ros.git && \
    cd .. && catkin_make && \
    sudo rm -rf /var/lib/apt/lists/*


# # Загрузка и установка Miniconda
# RUN /bin/bash /home/captain/visual-transformer/Miniconda3.sh -b -p $CONDA_DIR && \
#     $CONDA_DIR/bin/conda clean -tipsy
# Download base image ubuntu 16:04
# FROM ubuntu:16.04
FROM nvidia/cuda:10.1-base-ubuntu16.04

RUN apt-get update 
# Set locale fonts for compilation of exprimental apps
RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN mkdir /home/mltools/
COPY ./ /home/mltools/
WORKDIR /home/mltools/
RUN ls -ls /home/mltools/*

########################################################### ENVIRONMENT: base ################################################
RUN apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    software-properties-common \
    unzip

#######################################################Installation for OpenCV ###############################################
RUN apt-get update && \
    apt-get remove -y \
    x264 libx264-dev && \
    apt-get install -y \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    gfortran \
    libjpeg8-dev \
    libjasper-dev \
    libpng12-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev \
    libv4l-dev

RUN cd /usr/include/linux && \
      ln -s -f ../libv4l1-videodev.h videodev.h 

RUN apt-get install -y \
    libgstreamer0.10-dev \
    libgstreamer-plugins-base0.10-dev \
    libgtk2.0-dev \
    libtbb-dev \
    qt5-default \
    libatlas-base-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libavresample-dev \
    x264 \
    v4l-utils \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libeigen3-dev \
    libhdf5-dev \
    doxygen

##############################################################################################################################

# Install python 3.6
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv

# update pip
RUN python3.6 -m pip install pip --upgrade

######################################################### Application needed: ################################################
# Install 
RUN cd /home/mltools/

RUN python3.6 -m pip install -r requirements.txt

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

RUN cd /home/mltools/

RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout $cvVersion 

RUN curl -L https://github.com/opencv/opencv/archive/3.4.3.zip -o opencv.zip
RUN curl -L https://github.com/opencv/opencv_contrib/archive/3.4.3.zip -o opencvContrib.zip
RUN unzip -q opencvContrib.zip
RUN unzip -q opencv.zip && cd opencv-3.4.3/ && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_opencv_python3=yes \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ \
        -DOPENCV_EXTRA_MODULES_PATH=/home/mltools/opencv_contrib-3.4.3/modules \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3.6 \
        -DPYTHON3_INCLUDE=/usr/include/python3.6/ \
        -DPYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON3_LIBRARY=/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so \
        -DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.6/dist-packages/ \
        -DPYTHON_NUMPY_INCLUDE_DIR=/usr/local/lib/python3.6/dist-packageis/numpy/core/ \
        -DBUILD_NEW_PYTHON_SUPPORT=ON 
RUN cd /home/mltools/opencv-3.4.3/build && make -j 4 && make install && cd /home/mltools/  && rm opencv.zip && rm opencvContrib.zip && rm -rf opencv-3.4.3/ && rm -rf opencv_contrib-3.4.3/
      
RUN /bin/sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig


###############################################################################################################################

RUN python -m pip install torch

RUN python -m pip install torchvision


###############################################################################################################################

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /home/mltools/

RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/mltools as their home directory
ENV HOME=/home/mltools
RUN chmod -R 777 /home/mltools

ENTRYPOINT /bin/bash

# Copy Repo files
COPY ./ /home/mltools

# Move to the correct path
WORKDIR /home/mltools


# Entrypoint
ENTRYPOINT /bin/bash

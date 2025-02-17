FROM nvidia/cudagl:11.0-devel-ubuntu18.04 
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         htop \
         python3-dev \
         libopenblas-dev \
         libopenexr-dev \
         libcudnn7 \
         tmux \
         vim \
         rsync \
         unzip \
         wget \
         libgtk2.0-dev \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*

RUN  pip3 install --upgrade pip && \
     pip3 install setuptools && \
     pip3 install wheel && \
     pip3 install opencv-python matplotlib tqdm imgaug PyYAML scikit-learn easydict && \
     pip3 install open3d OpenEXR plyfile h5py && \
     pip3 install torch && \
     pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html

WORKDIR /workspace
RUN chmod -R a+w .

RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common && \
    apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
    add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u && \
    apt-get update && apt-get install -y --no-install-recommends \
         librealsense2-dkms \
         librealsense2-utils \
         librealsense2-dev \
         librealsense2-dbg

ADD https://api.github.com/repos/FlorisE/implicit_depth/git/refs/heads/main version.json
RUN git clone https://github.com/FlorisE/implicit_depth && \
    cd implicit_depth && \
    pip3 install -r requirements.txt

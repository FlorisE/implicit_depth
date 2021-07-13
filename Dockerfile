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
         libopenblas-dev \
         libopenexr-dev \
         libcudnn7 \
         tmux \
         vim \
         rsync \
         unzip \
         wget \
         libgtk2.0-dev && \
     rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch && \
     conda install -c conda-forge -c fvcore fvcore && \
     conda clean -ya
RUN  pip install --upgrade pip && \
     pip install opencv-python matplotlib tqdm imgaug PyYAML scikit-learn easydict && \
     pip install open3d OpenEXR plyfile h5py && \
     pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html

WORKDIR /workspace
RUN chmod -R a+w .

RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common
RUN apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN apt-get update && apt-get install -y --no-install-recommends \
         librealsense2-dkms \
         librealsense2-utils \
         librealsense2-dev \
         librealsense2-dbg

RUN conda create --name lidf python=3.7
RUN conda activate lidf

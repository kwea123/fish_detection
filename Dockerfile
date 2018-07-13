FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR "/root"

# install protobuf 3
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
RUN unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# install pycocotool for evaluation
RUN apt-get update
RUN apt install -y git python3-tk libsm6 libxext6
RUN pip3 install cython pillow opencv-python
RUN pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI




FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR "/root"

# install protobuf 3
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
RUN unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# install pycocotool for evaluation
RUN apt install git
RUN git clone https://github.com/pdollar/coco.git
RUN cd coco/PythonAPI
RUN make
RUN make install
RUN python setup.py install

RUN pip3 install keras pillow opencv-python==3.2.0.8 h5py



WORKDIR "/root"

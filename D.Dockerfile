FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt .

RUN apt update
RUN apt install git -y
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 libxext6 -y

RUN conda install Cython
RUN conda install -c menpo opencv
RUN pip install -r ./requirements.txt
RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


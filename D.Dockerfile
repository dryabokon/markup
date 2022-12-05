FROM continuumio/miniconda3

WORKDIR /app
 
COPY environment.yml .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -c menpo opencv
RUN conda env create -f environment.yml

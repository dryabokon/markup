FROM continuumio/miniconda3
# FROM python:3.9.15-bullseye

WORKDIR /app
 
COPY environment.yml .
COPY requirements.txt .

RUN apt-get update
RUN apt-get install libsm6 libxext6  -y
RUN conda install -c menpo opencv
# RUN conda env create -f environment.yml
RUN conda clean --all
RUN conda env update --file environment.yml  --prune
# RUN pip install -r ./requirements.txt

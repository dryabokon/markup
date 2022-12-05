FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN apt-get update
# RUN conda env update --file environment.yml --prune
RUN conda env create -n p39a -f environment.yml



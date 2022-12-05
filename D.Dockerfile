FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt .

RUN apt update
RUN apt install git -y
RUN apt install build-essential -y
RUN apt install libsm6 libxext6  -y

RUN conda install Cython
RUN conda install -c menpo opencv
RUN pip install -r ./requirements.txt
RUN pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# ---------------------------------------------------------------
# RUN conda install pycocotools
# RUN conda install -c "conda-forge/label/gcc7" pycocotools
# RUN conda env create -f environment.yml
# RUN conda clean --all
# RUN conda env update --file environment.yml  --prune




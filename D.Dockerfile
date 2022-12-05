FROM continuumio/miniconda3

WORKDIR /app
 
COPY environment.yml .
COPY requirements.txt .

RUN apt-get update
# RUN conda install -c "conda-forge/label/gcc7" pycocotools
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
RUN apt-get install libsm6 libxext6  -y
RUN conda install -c menpo opencv
RUN pip install -r ./requirements.txt

# RUN conda env create -f environment.yml
# RUN conda clean --all
# RUN conda env update --file environment.yml  --prune




docker_imge="p39-opencv-flask"

sudo docker run \
-v ${PWD}:/home \
-v ${PWD}/data:/home/data \
-v ${PWD}/tools:/home/tools \
--workdir /home \
-it $docker_imge \
python3 main_desktop.py




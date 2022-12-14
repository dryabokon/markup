# markup
Python backend services for annotation tool

## Client-Server architecture:
![alt text](/data/media/architecture.png)

## API:
![alt text](/data/media/api.png)

## Example of mask image:
![alt text](/data/media/000000193181.jpg)

## Installation for docker env

1. Clone sources and dependencies
```
git clone https://github.com/dryabokon/markup.git
cd markup
git clone https://github.com/dryabokon/tools.git
```

2. Build docker image
```
chmod +x ./docker_buildme.sh
sudo ./docker_buildme.sh
```

3. Run desktop app inside docker
```
./docker_runme.sh
```

## Installation for local virtual env

1. Clone sources and dependencies
```
git clone https://github.com/dryabokon/markup.git
cd markup
git clone https://github.com/dryabokon/tools.git
```

2. Install packages
```
pip install -r requirements.txt
```

3. Run desktop app
```
python3 main_desktop.py
```
The script will source an annotation file (./data/ex_02_coco_persons/person_keypoints_val2017.json)
and produce the mask image at the output folder (./data/output/)

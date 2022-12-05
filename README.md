# markup
Python backend services for annotation tool

## Client-Server architecture:
![alt text](/data/media/architecture.png)

## API:
![alt text](/data/media/api.png)

## Example of mask image:
![alt text](/data/media/000000193181.jpg)

## Installations steps: docker env

1. Clone dependencies
git clone https://github.com/dryabokon/tools.git

2. Build docker image
./buildme.sh

3. Run desktop app inside docker
./runme.sh

## Installations steps: local virtual env

1. Clone dependencies
git clone https://github.com/dryabokon/tools.git

2. Install packages
pip install -r requirements.txt

3. Run desktop app
python3 main_desktop.py
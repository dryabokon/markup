import json
import requests
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../tools')
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
host = 'http://localhost'
port = '8050'
# ----------------------------------------------------------------------------------------------------------------------
def load_annotation_json(filename_in):
    with open(filename_in, "r", encoding="utf-8") as f:
        dct_json = json.load(f)
    return dct_json
# ----------------------------------------------------------------------------------------------------------------------
def ex_get_healthcheck():
    api_namespace = '/healthcheck/v1/'
    payload_prefix = host + ':' + port + api_namespace

    print('GET ', payload_prefix)
    resp = requests.get(payload_prefix)
    print(resp.status_code)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_get_mask_image(dct_json):
    api_namespace = '/mask/v1/'
    payload_prefix = host + ':' + port + api_namespace

    print('GET ', payload_prefix)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    resp = requests.get(payload_prefix, json=dct_json, headers=headers)
    encoded_image = resp.json()['mask']
    image = tools_image.decode_base64(encoded_image)
    cv2.imwrite('./temp.png',image)
    print(resp.status_code)
    return
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_get_healthcheck()
    ex_get_mask_image(dct_json = load_annotation_json('./data/ex_01_coco_soccer/single.json'))



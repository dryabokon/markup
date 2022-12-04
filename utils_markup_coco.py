import numpy
import cv2
import os
import requests
from PIL import Image as PillowImage
from pycocotools.coco import COCO
import json
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../tools')
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_image
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class Markuper:
    def __init__(self, filename_coco_annotation_json,folder_images,folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_images=folder_images
        self.folder_out=folder_out
        self.col_bg = (250,250,250)

        self.coco = None
        self.import_JSON(filename_coco_annotation_json)

        self.colors = tools_draw_numpy.get_colors(16,colormap='tab10')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def reconvert_JSON(self,filename_in,filename_out,image_id=None,tabulate=True):

        with open(filename_in, "r", encoding="utf-8") as f:
            dct_json = json.load(f)

        if image_id is not None:
            if not isinstance(image_id, list):
                image_id = [image_id]
            dct_json['images'] = [im for im in dct_json['images'] if im['id'] in image_id]
            dct_json['annotations'] = [a for a in dct_json['annotations'] if a['image_id'] in image_id]

        with open(self.folder_out+filename_out, "w", encoding="utf-8") as f:
            json.dump(dct_json, f, indent=4 if tabulate else None,separators=(",", ":"))
        return
# ----------------------------------------------------------------------------------------------------------------------
    def import_JSON(self, filename_coco_annnotation_json):
        if filename_coco_annnotation_json is None:
            self.coco = COCO()
        elif isinstance(filename_coco_annnotation_json,str):
            self.coco = COCO(filename_coco_annnotation_json)
        else:
            self.coco = COCO()
            self.coco.dataset = filename_coco_annnotation_json
            self.coco.createIndex()

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_annotations_by_image_id(self, image_id):
        frame_annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        return frame_annotations
# ----------------------------------------------------------------------------------------------------------------------
    def download_image_by_URL(self, URL):

        image = None

        try:
            response = requests.get(URL, stream=True, timeout=2, allow_redirects=False)
        except:
            return image

        if not response.ok:
            return image

        try:
            image = cv2.cvtColor(numpy.array(PillowImage.open(response.raw)), cv2.COLOR_RGB2BGR)
        except:
            return image

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_bbox(self, frame_annotation, image, color):
        b = frame_annotation['bbox']
        if isinstance(b, list):
            box = b[1], b[0], b[1] + b[3], b[0] + b[2]
            image = tools_draw_numpy.draw_rect(image, box[1], box[0], box[3], box[2], color=color.tolist(), alpha_transp=0.8)
        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_contour(self, frame_annotation, image, color):
        for s in frame_annotation['segmentation']:
            if isinstance(s,list):
                image = tools_draw_numpy.draw_contours(image,numpy.array(s).reshape((-1, 2)),color=color.tolist(),w=2,transperency=0.75)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_annotations(self, download_images=False,skip_missing_images=True,skip_empty_annotations=True,save_to_disk=True,lim=50):
        tools_IO.remove_files(self.folder_out,list_of_masks='*.png,*.jpg,*.jpep',create=True)
        image = None
        cnt = 0
        for im in self.coco.dataset['images']:
            if cnt>lim:break
            image_id = im['id']
            filename_image = im['file_name']
            frame_annotations = self.get_annotations_by_image_id(image_id)
            if skip_empty_annotations and len(frame_annotations)==0:
                continue

            if self.folder_images is not None and os.path.isfile(self.folder_images+filename_image):
                image = cv2.imread(self.folder_images+filename_image)
            elif download_images:
                image = self.download_image_by_URL(im['coco_url'])
                if image is not None:
                    image = tools_image.desaturate(image,level=0.9)
            elif not skip_missing_images:
                image = numpy.full((im['height'], im['width'], 3), self.col_bg, dtype=numpy.uint8)
            else:
                continue

            for i,frame_annotation in enumerate(frame_annotations):
                image = self.draw_contour(frame_annotation, image,self.colors[i%len(self.colors)])
                #image = self.draw_bbox(frame_annotation, image, self.colors[i % len(self.colors)])

            if save_to_disk:
                cv2.imwrite(self.folder_out+filename_image,image)
            cnt+=1

        return image
# ----------------------------------------------------------------------------------------------------------------------



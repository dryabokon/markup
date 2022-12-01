import numpy
import cv2
import os

import pandas as pd
import requests
from PIL import Image as PillowImage
from pycocotools.coco import COCO
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
import tools_DF
import tools_image
import tools_IO
# ----------------------------------------------------------------------------------------------------------------------
class Markuper:
    def __init__(self, filename_coco_annotation_json, folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out=folder_out
        self.col_bg = (32,32,32)

        self.coco = None
        self.init_by_JSON(filename_coco_annotation_json)

        self.colors = tools_draw_numpy.get_colors(16,colormap='tab10')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_by_JSON(self, filename_coco_annnotation_json):
        self.coco = COCO(filename_coco_annnotation_json)

        self.df_images = pd.DataFrame({'id':[i['id'] for i in self.coco.dataset['images']],
                           'file_name':[i['file_name'] for i in self.coco.dataset['images']],
                           'coco_url': [i['coco_url'] for i in self.coco.dataset['images']],
                           'width': [i['width'] for i in self.coco.dataset['images']],
                           'height': [i['height'] for i in self.coco.dataset['images']]
                           })

        self.df_annotations = pd.DataFrame({'id':[a['id'] for a in self.coco.dataset['annotations']],
                                       'image_id': [a['image_id'] for a in self.coco.dataset['annotations']],
                                       'record_id':range(len(self.coco.dataset['annotations']))
                                       })

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_annotations_by_image_id(self, image_id):
        df_annotations_f = tools_DF.apply_filter(self.df_annotations, 'image_id', image_id)
        frame_annotations = [self.coco.dataset['annotations'][r] for r in df_annotations_f['record_id'].values]
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
    def draw_bboxes(self, frame_annotation, image,color):
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
    def process_local_folder(self, folder_images):

        filename_images = tools_IO.get_filenames(folder_images,'*.*')

        df = self.df_images[self.df_images['file_name'].isin(filename_images)]

        for r in range(df.shape[0]):
            filename_image = df.iloc[r]['file_name']
            image = cv2.imread(folder_images+filename_image)
            image = tools_image.desaturate(image, level=0.9)
            image_id = df.iloc[r]['id']
            frame_annotations = self.get_annotations_by_image_id(image_id)
            for i,frame_annotation in enumerate(frame_annotations):
                image = self.draw_contour(frame_annotation, image,self.colors[i%len(self.colors)])

            cv2.imwrite(self.folder_out + filename_image, image)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def process_remote_storage(self,download_images=False,lim=50):
        tools_IO.remove_files(self.folder_out,create=True)

        for r in range(self.df_images.shape[0]):
            if r>lim:break
            image_id = self.df_images.iloc[r]['id']
            filename_image = self.df_images.iloc[r]['file_name']
            frame_annotations = self.get_annotations_by_image_id(image_id)
            if len(frame_annotations)==0:
                continue

            if download_images:
                URL = self.df_images.iloc[r]['coco_url']
                image = self.download_image_by_URL(URL)
                if image is not None:
                    image = tools_image.desaturate(image,level=0.9)
            else:
                W = self.df_images.iloc[r]['width']
                H = self.df_images.iloc[r]['height']
                image = numpy.full((H, W, 3), self.col_bg, dtype=numpy.uint8)

            for i,frame_annotation in enumerate(frame_annotations):
                image = self.draw_contour(frame_annotation, image,self.colors[i%len(self.colors)])
                #image = self.draw_bboxes(frame_annotation, image,self.colors[i%len(self.colors)])

            cv2.imwrite(self.folder_out+filename_image,image)


        return
# ----------------------------------------------------------------------------------------------------------------------


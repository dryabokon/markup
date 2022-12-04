from flask import Flask, request, json, jsonify
from flask_restx import Resource, Api, reqparse
# ----------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
api = Api(app,doc='/')
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../tools/')
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
import utils_markup_coco
M = utils_markup_coco.Markuper(None,folder_images=None,folder_out='./')
# ----------------------------------------------------------------------------------------------------------------------
namespace_healthcheck = api.namespace('healthcheck/v1/', description='Healthcheck API')
@namespace_healthcheck.route('/')
class Healthcheck(Resource):
    def get(self):
        return 'OK'

    def put(self,value):
        return value
# ----------------------------------------------------------------------------------------------------------------------
namespace_mask = api.namespace('mask/v1/', description='Creation of mask from COCO annotation')
@namespace_mask.route('/')
class Mask(Resource):
    def get(self):
        dct_json = json.loads(request.data)
        M.import_JSON(dct_json)
        image = M.draw_annotations(download_images=False, skip_missing_images=False, skip_empty_annotations=False,save_to_disk=False, lim=1)
        encoded_image_str = tools_image.encode_base64(image).decode("ascii")
        dct_json_response = jsonify({'mask': encoded_image_str})
        return dct_json_response
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    app.run(debug=False, host='localhost', port=8050)



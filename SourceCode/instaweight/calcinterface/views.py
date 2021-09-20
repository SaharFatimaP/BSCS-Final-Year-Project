from django.shortcuts import render, HttpResponse
import numpy as np
import json
from utils import weight_utils, weight_calculator, smoothing_utils
from matplotlib import pyplot as plt
from dashboard.models import Cattle, DailyWeight
from .apps import CalcinterfaceConfig
import json
import cv2
# Create your views here.


def index(request, id):
    cattle = Cattle.objects.get(id=id)
    if request.method == 'POST':
        print(request.FILES['image'], request.POST)
        depth_image = np.load(request.FILES['depth-image'])

        hg_raw = get_hg_vector(request, depth_image)
        hg_raw = smoothing_utils.remove_outliers(hg_raw)

        heart_girth_raw = weight_utils.length_of_depth_vec(hg_raw)*39.37*2

        hg_smoothed = smoothing_utils.smooth_using_peaks(hg_raw)

        heart_girth_smoothed = weight_utils.length_of_depth_vec(hg_smoothed)*39.37*2

        weight_raw = weight_calculator.using_hg(heart_girth_raw)
        weight_smoothed = weight_calculator.using_hg(heart_girth_smoothed)

        # bl_raw = get_bl_vector(request, depth_image)
        # bl_smoothed = smoothing_utils.smooth_using_peaks(bl_raw)
        # body_length_raw = weight_utils.length_of_depth_vec(bl_raw)*39.37
        # ody_length_smoothed = weight_utils.length_of_depth_vec(bl_smoothed)*39.37

        # weight_using_bl = weight_calculator.using_bl(heart_girth_smoothed, body_length_smoothed)

        context = {
            'cattle': cattle,
            'heart_gith_raw': heart_girth_raw,
            'heart_girth_smoothed': heart_girth_smoothed,
            # 'body_length_raw': body_length_raw,
            # 'body_length_smoothed': body_length_smoothed,
            'weight_using_raw_data': weight_raw,
            'weight_using_smoothed_data': weight_smoothed,
            # 'weight_using_smoothed_bl': weight_using_bl,
        }
        return render(request, "calcinterface/response.html", context=context)

    context = {
        'cattle': cattle,
    }
    return render(request, "calcinterface/index.html", context=context)


def predict_box(request):
    print(request.FILES['image'], request.POST)
    image = request.FILES['image']
    r_image, ObjectsList = CalcinterfaceConfig.yolo.detect_img(
        np.fromstring(image.read(), np.uint8))
    print(ObjectsList)
    if len(ObjectsList):
        predictions = [int(i) for i in ObjectsList[0][0:5]]
        cv2.imwrite("img.png", r_image)
        return HttpResponse(json.dumps({
            'hgTop': predictions[0:2],
            'hgBottom': [predictions[0], predictions[3]]
        }))


def get_hg_vector(request, depth_image):
    points = json.loads(request.POST['points'])
    hg_top = points['hgTop']
    hg_bottom = points['hgBottom']
    return depth_image[hg_top[1]:hg_bottom[1], hg_top[0]]


def get_bl_vector(request, depth_image):
    points = json.loads(request.POST['points'])
    front_shoulder = points['frontShoulder']
    pin_bone = points['pinBone']
    return weight_utils.body_length_depth_vec(front_shoulder, pin_bone, depth_image)

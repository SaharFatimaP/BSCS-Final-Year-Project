from YOLO.image_detect import YOLO
from django.apps import AppConfig


class CalcinterfaceConfig(AppConfig):
    name = 'calcinterface'
    yolo = YOLO()

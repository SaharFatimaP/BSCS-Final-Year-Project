from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register([
    Cattle,
    Breed,
    DailyWeight,
    Status,
    AgeRange,
    StandardWeightCriteria,
    Alert,
    CattleStatus
])
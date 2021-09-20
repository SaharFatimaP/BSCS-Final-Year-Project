from dashboard.models import *
from datetime import datetime



def weight_distribution():
    distribution ={
        'over_weight':0,
        'under_weight':0,
        'normal':0
    }

    cattle_objs = Cattle.objects.all()
    for cattle in cattle_objs:
        weight = DailyWeight.objects.all().order_by('date_time').last()
        birth_date = cattle.birth_date
        today = datetime.now().date()
        age_months = (today.year - birth_date.year)*12 
        age_range = AgeRange.objects.filter(start_range__gte=age_months,end_range__lte=age_months+6)
        print(age_months, age_range)



    return distribution 






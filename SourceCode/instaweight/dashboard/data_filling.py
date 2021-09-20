from dashboard.models import *
from datetime import datetime, timedelta
import random 
import string

def breed():
    breed_id = random.randint(1,1000)
    breed_name = "Holstein"
    breed_obj = Breed(breed_id, breed_name)
    breed_obj.save()
    return breed_obj

def age_range():
    cattle_lst = []
    ranges = ["1-6","7-12","13-18","19-24","25-30", "31-36", "37-42", "43-48"]
    for rang in ranges:  
        lst = rang.split("-")
        start_range = int(lst[0])
        end_range = int(lst[1])
        age_range_obj = AgeRange(start_range=start_range, end_range=end_range)
        age_range_obj.save()
        cattle_lst.append(age_range_obj)
    return cattle_lst

def cattle(breed_obj):
    cattle_lst = [] 
    date_count = 1
    for our_cattle in range(200):
        #status = random.choice(["Under Weight", "Standard", "Over Weight"])
        gender = random.choice(["Male", "Female"])
        rf_id = str(datetime.now()) + str(random.randint(1, 1000))
        ear_tag = str(datetime.now()) + str(random.randint(1, 1000))
        today_date = datetime.now().date()
        birth_date = today_date + timedelta(days = 7*date_count)
        cattle_obj = Cattle(breed=breed_obj, gender=gender, rf_id=rf_id, ear_tag=ear_tag, birth_date=birth_date)
        cattle_obj.save()
        cattle_lst.append(cattle_obj)
        date_count += 1
    return cattle_lst

def daily_weight(cattle_objs):
    weight_lst = []
    for cattle in cattle_objs:
        date_time = datetime.now()
        heart_girth = random.randint(65, 205)
        diagonal_len = random.randint(230, 260)
        weight = (diagonal_len + (heart_girth)**2)/10840
        image ='https://picsum.photos/600'
        depth_img = 'https://picsum.photos/600'
        daily_weight_obj = DailyWeight(cattle=cattle, heart_girth=heart_girth, diagonal_len=diagonal_len, weight=weight, image=image, depth_img=depth_img)
        daily_weight_obj.save()
        weight_lst.append(daily_weight_obj)
    return weight_lst



def standard_criteria(breed_obj,age_range_obj):
    cattle_lst = []
    weight_ranges = ["56-160","184-292","331-400","401-496","497-564","565-650","650-725","726-800"]
    for i, weight_range in enumerate(weight_ranges):
        lst = weight_range.split("-")
        from_weight = int(lst[0])
        to_weight = int(lst[1])
        std_weight_obj = StandardWeightCriteria(breed=breed_obj, age_range=age_range_obj[i], from_weight=from_weight, to_weight=to_weight)
        std_weight_obj.save()
        cattle_lst.append(std_weight_obj)
    return cattle_lst


StandardWeightCriteria.objects.all().delete()
DailyWeight.objects.all().delete()
Cattle.objects.all().delete()
Breed.objects.all().delete()
AgeRange.objects.all().delete()


print("Execution start")
breed_obj = breed()
print(f"Breed Created #: {breed_obj}")
age_range_objs = age_range()
print(f"Age Ranges Created #: {len(age_range_objs)}")
cattle_objs = cattle(breed_obj)
print(f"Cattls Created #: {len(cattle_objs)}")
cattle_weights = daily_weight(cattle_objs)
print(f"Cattl Weights Created: {len(cattle_weights)}")
standard_criteria(breed_obj,age_range_objs)
print("Execution sucessfull .... ok")



















        
        
    


    


        




    
    
    

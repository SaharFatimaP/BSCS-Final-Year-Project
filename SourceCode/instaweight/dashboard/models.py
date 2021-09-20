from django.db import models
from django.db.models import Count, Q
from django.utils import timezone
from datetime import datetime, date


# Create your models here.

# * Have to discuss "can we delete a cattle's record, change status"


class Breed(models.Model):
    '''
    Represents a bread
    '''
    breed_name = models.CharField(max_length=250)

    @property
    def Cattle_Count(self):
        return self.cattle_set.all().count()

    def __str__(self):
        return self.breed_name


class Status(models.Model):
    '''
    Represents a generic status. i.e. "Alive", "Dead", "Undertreatment"
    '''
    status_value = models.CharField(max_length=250)

    def __str__(self):
        return self.status_value


class Cattle(models.Model):
    ''''''
    breed = models.ForeignKey(Breed, on_delete=models.PROTECT)
    gender = models.CharField(
        choices=[("Male", "Male"), ("Female", "Female")], max_length=50, verbose_name="Gender")
    rf_id = models.CharField(max_length=50, verbose_name="RF ID", unique=True)
    ear_tag = models.CharField(
        max_length=50, verbose_name="Ear ID", unique=True)
    birth_date = models.DateField(verbose_name="Birth Date")
    image = models.ImageField(verbose_name='Image')

    def __str__(self):
        return f'{self.ear_tag} | {self.breed}'

    @classmethod
    def gender_distribbution(cls):
        return {
            "male": cls.objects.filter(gender='M').count(),
            "female": cls.objects.filter(gender='F').count(),
        }

    @property
    def Age(self):
        today = date.today()
        return today.year - self.birth_date.year - ((self.birth_date.month, today.day) < (self.birth_date.month, self.birth_date.day))

    @property
    def Status(self):
        status = self.status_objects.all().order_by('-date_time').first()
        print(status)
        return status

    def __str__(self):
        return f'{self.ear_tag} | {self.breed} | {self.Status}'


class CattleStatus(models.Model):
    status = models.ForeignKey(Status, on_delete=models.PROTECT)
    cattle = models.ForeignKey(Cattle, on_delete=models.PROTECT, related_name='status_objects')
    date_time = models.DateTimeField(auto_now=True, verbose_name='Date Time')

    def __str__(self):
        return self.status.__str__()


class AgeRange(models.Model):
    ''''''
    start_range = models.IntegerField(verbose_name="Starting Age Range")
    end_range = models.IntegerField(verbose_name="Ending Age Range")

    class Meta:
        unique_together = (('start_range', 'end_range'),)

    def __str__(self):
        return f'from: {self.start_range} to: {self.end_range}'


class Alert(models.Model):
    cattle = models.ForeignKey(Cattle, on_delete=models.CASCADE)
    message = models.TextField(verbose_name='Alert Message')
    date_time = models.DateTimeField(auto_now=True, verbose_name='Date Time')

    def __str__(self):
        return self.message[:20]


class DailyWeight(models.Model):
    date_time = models.DateTimeField(verbose_name='Date Time', auto_now=True)
    cattle = models.ForeignKey(Cattle, on_delete=models.CASCADE, related_name='logged_weights')
    weight = models.FloatField(verbose_name='Weight in KGs')
    heart_girth = models.FloatField(verbose_name='Heart Girth in meters', blank=True)
    diagonal_len = models.FloatField(verbose_name='Diagonal Length in meters', blank=True)
    image = models.URLField(verbose_name='Image', blank=True)
    depth_img = models.URLField(verbose_name='Depth Image', blank=True)

    def __str__(self):
        return f"Cattle #{self.cattle.id}'s weight on {self.date_time} is {self.weight}"


class StandardWeightCriteria(models.Model):
    age_range = models.ForeignKey(AgeRange, on_delete=models.PROTECT)
    breed = models.ForeignKey(Breed, on_delete=models.PROTECT)
    from_weight = models.FloatField(verbose_name='Starting Weight in KGs')
    to_weight = models.FloatField(verbose_name='Ending Weight in KGs')


# Models to keep track of change history are remaining

from calcinterface import views
from django.urls import path

urlpatterns = [
    path('/<int:id>', views.index, name='weight-calculator'),
    path('/predict', views.predict_box, name='predict'),
]

from django.urls import path

from . import views


urlpatterns = [
    path('login', views.login, name='login'),
    path('', views.dashboard, name='dashboard'),
    path('cattle-management', views.cattle_management, name='cattle-management'),
    path('cattle_view/<int:id>', views.cattle_view, name='cattle_view'),
    path('daily_view/<int:id>', views.daily_view, name='daily_view'),
    path('settings', views.settings, name='settings'),
    path('form', views.cattle_form, name="cattle_form"),
    path('log-weight/<int:id>', views.log_weight, name="log_weight")
    
]
   

from django.urls import path

from . import views

app_name='nlp'

urlpatterns = [
    path('',views.homePage,name='home'),
    path('process/',views.process,name='process'),
]
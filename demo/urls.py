from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^plot_test1/$', views.plot_test1, name='plot_test1'),
    url(r'^plot_test2/$', views.plot_test2, name='plot_test2'),
    url(r'^$', views.demo_home, name='demo_home')
]

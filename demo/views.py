from django.shortcuts import render, render_to_response
from django.http import HttpResponse

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import mpld3
import json

"""
load machine learning / deep learning model
"""
from demo.fizzbuzz import PredictService

checkpoint_path = "./ckpt_dir"
checkpoint_file = "./ckpt_dir/model.ckpt-3100.meta"
predict_service = PredictService(checkpoint_path, checkpoint_file)


def demo_home(request):
    """ Default view for the root """
    return render(request, 'demo/home.html')

def plot_test1(request):
    context = {}
    fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
    N = 100

    """
    Demo about using matplotlib and mpld3 to rendor charts
    """
    scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s=1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)
    ax.grid(color='white', linestyle='solid')

    ax.set_title("Scatter Plot (with tooltips!)", size=20)

    labels = ['point {0}'.format(i + 1) for i in range(N)]
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)
    #figure = mpld3.fig_to_html(fig)
    figure = json.dumps(mpld3.fig_to_dict(fig))
    context.update({ 'figure' : figure })
    
    """
    Demo about using tensorflow to predict the result
    """
    num = np.random.randint(100)
    prediction = predict_service.predict(num)     
    context.update({ 'num' : num })
    context.update({ 'prediction' : prediction })
    
    return render(request, 'demo/demo.html', context)


def plot_test2(request):
    fig = plt.figure()
    plt.plot([1,2,3,4])
    g = mpld3.fig_to_html(fig)
    return HttpResponse(g)
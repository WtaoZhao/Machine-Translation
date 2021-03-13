from django.shortcuts import render, redirect
from backend.translate import Translator

def redirectToHome(request):
    return redirect('/nmtsys/')

def home(request, ctx={}):
    return render(request, 'frontend/home.html', ctx)

def translate(request):
    source = request.POST['source']
    src = request.POST['src']
    dst = request.POST['dst']

    if src == dst:
        target = source
    else:
        # only support ne -> zh on March 13th, 2021
        translator = Translator(src, dst)
        target = translator.translate([source])

    ctx = {'source': source, 'target': target, 'src': src, 'dst': dst}
    return home(request, ctx)

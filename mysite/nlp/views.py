from django.shortcuts import render

# Create your views here.

def homePage(request):
    return render(request,'nlp/home.html')

def process(request):
    # contents=request.POST['contents']
    contents=request.POST['contents2']
    src=request.POST['src']
    dst=request.POST['dst']
    results=src+'?'+dst+'?'+contents
    return render(request,'nlp/home.html',{'left':contents, 'right':results, 'src':src, 'dst':dst })

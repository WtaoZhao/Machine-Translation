## Low-resource Machine Translation System Demo
Update on March 13th, 2021
#### Introduction
* **frontend/**:  a Django app serving as web frontend
* **backend/**:  a translation module called by frontend	
	* Note:  The trained model is stored in **backend/model/**, and currently we only support translation from Nepali to Chinese (*NMT.ne-zh.pth*). Later we will add another 5 models here.
#### Requirements
PyTorch &  Jieba
#### Run

```shell
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```



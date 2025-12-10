Nalog.ru капча тренировка и решение
основано на https://www.kaggle.com/code/tuqayabdullazade/captcha-recognition-with-tensorflow-90/notebook

Зависимости:
для тренировки
pip install imgaug==0.4.0
pip install opencv-python==4.10.0.84
pip install tensorflow==2.10.0
pip install numpy==1.26.4

под 4070 установить, под другие видеокарты нужно смотреть версию по таблице cuda/видеокарта
Nvidia CUDA Development 11.8
Nvidia CUDA Nsight 11.8
Nvidia CUDA Runtime 11.8

Запуск:
1 - разметить файлы, назвать их содержимым, положить в c:\prj\fns-captcha\pics\*.jpg, вид файла 045627.jpg
2 - запустить egdes.py и будет создан каталог png с файлами почищенными в формате png
3 - запустить тренировку train.py на основе png файлов
4 - модель будет сохранена в models\model.h5
5 - запустить проверку predict.py

# Kontur-test-task-
Тестовое задание направленное на распознание AI картинок, выполненное в качестве соревнования на kaggle 

Выполнение тестового задания
— Оглавление —
* О решении
* Документация
* Методы использованные при обучении 
* Данные 

### О решении
Основное решение сдано в виде двух python скриптов. Всё разбито по классам, каждый из которых позволяет гибко работать с кодом и модифицировать его, добавляя функционал. Такая реализация позволяет использовать этот код, если не в прод, то хотя бы в качестве прототипа. 
Веса моделей хранятся на HF
Документация
К сожалению, комментариями я покрыл только половину кода, поэтому помимо описания процесса запуска кода, я также опишу зачем нужен каждый класс и какой функционал, несет в себе каждая функция. 

### Запуск кода
Перед запуском производится авторизация на HFApi. Это необходимо для возможности загрузить веса уже обученной модели, а также позволяет сохранять модели на hugginngface

>>>pip install -U "huggingface_hub[cli]"
>>>huggingface-cli login

Подробнее здесь: https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login

Код запускается в файле main.py. В нем инициализируются экземпляры классов: 
Config,
ImageProcessor,
LabelManager,
Model,
Train,
Predict

Также происходит загрузка датасета 
PATH_TO_DATA = ""
dataset = load_dataset("imagefolder", data_dir=PATH_TO_DATA)
В папке с данными должна быть следующая структура:
 папка/ 
ai 
real 


В solution_kontur.py в начале импортируются неободимые библиотеки, а также определяются глобальные переменные: 
DATA_PATH - путь до данных
MODEL_CHECKPOIT - предобученная модель с HF. 
По умолчанию = "microsoft/swin-tiny-patch4-window7-224"
DEFAULT_NAME - то что будет стоять в конце названия модели. 
По умолчанию = '1st_try_w_classes'
Класс Config
 	Задает параметры для обучения модели 
update_params позволяет задавать параметры в виде словаря.

### Класс ImageProcessor 
Подготавливает картинки к обучению и валидации. На вход получает config, dataset.
image_processor импортируется из библиотеки transformer. В коде можно заменить "microsoft/swin-tiny-patch4-window7-224" на MODEL_CHECKPOINT и таким образом изображения буду подготавливаться к обучению с учетом используемой архитектуры. 
		
get_img_processor - возвращает название препроцессора
	
_normalize - Проводит аугментацию над батчами картинок. Среди аугментаций: RandomResizedCrop(crop_size), RandomHorizontalFlip(), ToTensor(), normalize.
	
_preprocess_train и _preprocess_val применяет _normalize к батчам картинок 

split_and_transform - Делит данные на train и validation. Возвращает готовые данные для передачи в Trainer.


### Класс LabelManager 
Создает два словаря для перевода меток в 0 и 1

make_label - Создает маппинги из списка меток.

check_label - Проверяет корректность маппинга между id и метками.


### Класс Model 
Создает модель на основе config.model_checkpoint. Все параметры берутся из config.

Класс Train
_collate_fn - функция, которая является пользовательской функцией сбора данных (collate function) для подготовки батчей данных
	
train - обучает, модель
 
evaluate - проводит валидацию 

push_to_hub - отправляет модель на hf 

### Класс Predict
Позволяет выполнять предикт, не только на обученный нами модели, но и просто загрузить модель с HF.

_data_preprocess - принимает на вход pandas.DataFrame. Возвращает кортеж из исходного датафрейма и датасета 


predict - принимает на вход pandas.DataFrame. Возвращает исходный датафрейм с колонкой `target`


set_device - смена устройства: GPU/CPU


show_device - актуальное устройство 


set_repo_name - позволяет загрузить любую предобученную модель с HF. С одним НО! в коде при инициализации используется строка:
self.repo_name = 't1msan/' + config.default_repo_name
на этапе написания кода использовались только обученные/дообученнные мной модели поэтому указан мой репозиторий на HF. Исправить для использования любых других моделей.


### Методы использованные при обучении (сново attention is all…)
В поисках современных методов работы с картинками в задаче классификации большинство дорог так или иначе ведут к тарнсформерам.   
Использовал:
VIT: 	 timm/vit_base_patch16_clip_384.laion2b_ft_in1k (предобученная)
SWIN: microsoft/swin-tiny-patch4-window7-224
	 microsoft/swin-base-patch4-window7-224-in22k (предобученная)
 microsoft/swinv2-tiny-patch4-window8-256   
Не трансформер, но используются идеи из swin - ConvNeXT 
 facebook/convnext-large-384-22k-1k
### Данные 	 

Данные обогащались из следующих источников, но конечно не всё и сразу. 
https://huggingface.co/datasets/InfImagine/FakeImageDataset/tree/main  (SD1.5  SD2.1  Midj5  StyLegan3)
https://huggingface.co/datasets/InfImagine/FakeImageDataset/tree/main/ImageData/train/stylegan3-80K  (StyLegan3)
https://www.kaggle.com/code/lifeofcoding/ai-image-detection-tensorflow/input?select=RealArt (Настоящие и нейросетевые картинки
https://www.kaggle.com/datasets/adityajn105/flickr30k?select= (Настоящие картинки)
https://huggingface.co/datasets/Bingsu/Human_Action_Recognition/tree/main/data   (Настоящие картинки)

Изначально были собраны следующие датасеты, разного размера и разного наполнения, такие датасеты хороши для любых картинок, но не для конкретного теста.
>>>kaggle datasets download -d don121/train-15k
>>>kaggle datasets download -d don121/train-1-2k
>>>kaggle datasets download -d don121/midj-sample-6k
>>>kaggle datasets download -d don121/52k-ai-and-real-img

В тесте из картинок с очевидными ai метки можно поставить картинкам со стилизованными собачками, картинки с искаженными лицами, идеализированные картинки с неестественным светом, гитаристы и строители.






Для достижения наилучшего результата, я отобрал из датасетов похожие картинки и обогатил ими датасет. 
Лица и природу в основном берем из SDv2
https://huggingface.co/datasets/InfImagine/FakeImageDataset/blob/main/ImageData/train/SDv15R-CC1M/SDv15R-dpmsolver-25-1M/SDv15R-CC1M.tar.gz.000
Собачек и и стилизованные картинки из Midjourneyv5
https://huggingface.co/datasets/InfImagine/FakeImageDataset/blob/main/ImageData/val/Midjourneyv5-5K/Midjourneyv5-5K.tar.gz


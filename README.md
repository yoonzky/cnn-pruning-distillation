# cnn-pruning-distillation

Свёрточная сеть на CIFAR-10, её сжатие итеративным прунингом и попытка дистилляции в рекуррентного ученика.

[![CNN в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoonzky/cnn-pruning-distillation/blob/main/cnn_cifar10.ipynb)
[![Прунинг в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoonzky/cnn-pruning-distillation/blob/main/pruning_distillation.ipynb)

## cnn_cifar10.ipynb

Классификатор CIFAR-10: четыре блока Conv — BatchNorm — Conv — BatchNorm — MaxPool — Dropout с числом фильтров 32, 64, 128, 256 и картами 32x32, 16x16, 8x8, 4x4, дальше flatten и Dense. Adam, learning rate 0,001, 25 эпох, batch 64.

Результат — **84,24 %** на тестовом наборе.

## pruning_distillation.ipynb

**Прунинг.** Пять итераций с растущим порогом: веса ниже порога обнуляются, затем одна эпоха дообучения (на последней итерации — пять).

| Порог | 0,05 | 0,10 | 0,15 | 0,20 | 0,25 |
| --- | --- | --- | --- | --- | --- |
| Точность | 84,14 % | 84,29 % | 81,94 % | 81,69 % | **81,63 %** |

До порога 0,10 качество не падает вовсе, дальше теряется около 2,5 процентных пункта.

**Дистилляция.** Класс `Distiller` с температурой и двумя слагаемыми лосса; ученик — один слой LSTM плюс один Dense, картинка подаётся ему как 32 шага по 96 признаков, то есть построчно. Перебраны тридцать конфигураций: LSTM из 512, 256, 128, 64, 32, 16 и Dense из 128, 64, 32, 16, 10.

Цель "ученик не хуже учителя" не достигнута: лучший результат — около 55,7 % против 85 % у учителя, часть конфигураций вырождается в предсказание одного класса. Причина в самом ученике: читая картинку построчно, LSTM теряет пространственные связи, которые свёртка видит сразу.

## Запуск

```
pip install -r requirements.txt
jupyter notebook cnn_cifar10.ipynb
```

CIFAR-10 качается первой ячейкой с зеркала на Hugging Face и разбирается в тот же формат, что отдаёт `keras.datasets.cifar10.load_data()`: около 15 секунд вместо четверти часа с сайта Торонто. Запасной путь — оригинальный архив оттуда же. Файлы кладутся в `./cifar10_data` и при следующем запуске берутся с диска.

## Стек

Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, seaborn; pyarrow и Pillow — для чтения датасета.

Учебный проект курса «Введение в разработку систем искусственного интеллекта», СПбГЭТУ «ЛЭТИ», 2025.

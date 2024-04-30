# Сжатие bmp24 изображений с использованием SVD

## Описание
В работе представлено 3 различных алгоритма сингулярного разложения:

- Numpy SVD
- [SVD Power method](http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf)
- [Block SVD Power Method](https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html#j_jisys-2018-0034_fig_004)

## Эксперимент

### Градиент
Для начала попробуем сжать в 5 раз картинку "градиент".
Посмотрим на то, как алгоритмы справляются с плавно переходящими цветами.

| Оригинал                                        | Numpy                                              | Power                                         | Block power                                         |
|-------------------------------------------------|----------------------------------------------------|-----------------------------------------------|-----------------------------------------------------|
| <img src="gradient/original.bmp" width="250px"> | <img src="gradient/numpy.bmp" width="250px">       | <img src="gradient/power.bmp" width="250px">  | <img src="gradient/block_power.bmp" width="250px">  |

В результате получаем 4 одинаковые (или почти одинаковые) картинки. С плавно переходящими цветами алгоритмы справились отлично.

### Геометрические фигуры

Теперь попробуем сжать картинку с геометрическими фигурами (на не есть и резкие цветовые переходы).

| Оригинал                                      | Numpy                                             | Power                                          | Block power                                          |
|-----------------------------------------------|---------------------------------------------------|------------------------------------------------|------------------------------------------------------|
| <img src="shapes/original.bmp" width="250px"> | <img src="shapes/numpy.bmp" width="250px">        | <img src="shapes/power.bmp" width="250px">     | <img src="shapes/block_power.bmp" width="250px">     |

С более резкими переходами в цветах все три алгоритма справляются уже не так хорошо. 

Важно заметить, что треугольники и квадраты имеют более четкое очертание, чем круги. (Круги стали похожи на квадраты :) )

Также важно заметить, что алгоритмам Numpy и Block SVD Power лучше удалось сохранить четкость фигур.

### Черно-белая фотография

Теперь попробуем сжать черно-белую фотографию Тимы в x5.

| Оригинал                                    | Numpy                                       | Power                                       | Block power                                       |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="Tima/original.bmp" width="250px"> | <img src="Tima/x5/numpy.bmp" width="250px"> | <img src="Tima/x5/power.bmp" width="250px"> | <img src="Tima/x5/block_power.bmp" width="250px"> |

Алгоритмы довольно хорошо справились сохранить красивую плитку на фоне, однако сохранить в хорошем качестве Тиму им не удалось (опять-таки из-за резких изменений цвета).

Тут также стоит заметить, что алгоритмы Numpy и Block SVD Power справились лучше (можно посмотреть, например, на глаза)

<details>
<summary>Интересный факт</summary>

Если попробовать сжать фотографию Тимы в x1 раз, то получим следующий результат:

| Оригинал                                    | Numpy                                       | Power                                       | Block power                                       |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="Tima/original.bmp" width="250px"> | <img src="Tima/x1/numpy.bmp" width="250px"> | <img src="Tima/x1/power.bmp" width="250px"> | <img src="Tima/x1/block_power.bmp" width="250px"> |

При детальном рассмотрении фотографий мне удалось выяснить, что алгоритм SVD Power оставляет меньше артефактов, чем два остальных алгоритма.
Это интересно, поскольку в двух предыдущий показательных экспериментах  SVD Power показывал худший результат.

</details>

## Вывод

- Все три алгоритма показывают довольно неплохой результат даже при сильном сжатии картинки. Однако Numpy и Block SVD Power все же справляются лучше.
- У данных SVD алгоритмов есть проблемы со сжатием кругов и, вероятно, со всеми фигурами, имющими большое кол-во углов.
- Алгоритмы отлично справляются с картинками, на которых смена цветов происходит плавно.
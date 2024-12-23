# GeoGarModel
Software package for geological and geophysical modeling of ground and aerial survey data

# Главное окно программы:
Главное окно программы представлено пространством для визуализации грид файлов регулярной и нерегулярной сети формата *zmap. Пример файла формата *.zmap можно получить путем экспорта грид файла в формат *.zmap в программном обеспечении Surfer 16. Примеры файлов Zmap также представлены в главном дистрибутиве программы.
В правой части располагается список загруженных файлов формата *.zmap. При нажатии на соответствующий файл будет происходить его визуализация в левом окошке со стандартными элементами оформления, а именно:
 - цветовая шкала rainbow
- гистограммное цветовое распределение
- без названия карты
- без подписи цветовой шкалы
Под списком загруженных файлов располагаются элементы изменения оформления карты, которые включают в себя:
- Минимальное значение цветового отображения
- Максимальное значение цветового отображения
- Цветовая шкала:
  -hsv
  -bwr
  -colorwarm
  -gist_rainbow
  -ocean
  -rainbow
  -seismic
  -tab20c
  -terrain
  -twilight
  -twilight_shifted
  
![image](https://github.com/user-attachments/assets/35c4becd-260b-4b18-a758-65778f58f4cf)
- Название карты
- Подпись цветовой шкалы (справа от цветовой шкалы; rotate = 90)
Под элементами оформления карты находятся 3 кнопки:
1)	«Построить точки обучающей и тестовой выборки». После загрузки обучающего СС датасета, точки обучающей и контрольной (валидационной) выборок можно визуализировать на карте. !Система координат (СК) отображающегося *.zmap файла должна совпадать с СК обучающего датасета для корректной визуализации.
2)	«Добавить shp файл». После загрузки одного или нескольких файлов формата *.shp их можно визуализировать на карте. !Система координат (СК) отображающегося *.zmap файла должна совпадать с СК файла формата *.shp для корректной визуализации.
3)	Кнопка «ОК». При нажатии данной кнопки для уже отображающегося файла формата *.zmap обновляются его элементы оформления. !При этом пункты 1 и 2 полностью удаляются с карты, но сами файлы сохраняются в корне программы и повторная их загрузка не требуется.

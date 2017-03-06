# Cat & Dog Image Classification
This project is for classify image for cat and dog

#### Requirement:
- Tensorflow deeplearning framework
- numpy

#### Dataset
"dataset" folder contain images about cat and dog
    - the format of the files is "cat_$number.jpg" and "dog_$number.jpg" the $number is number as index
    - ground-thruth is csv file("label.csv") with following format:
        - header (filename, label)
        - content (filename, label = cat, dog)

dataset can be download from https://www.kaggle.com/c/dogs-vs-cats

#### Change dataset into tensor
we follow this tutorial for changing image into tensor https://gist.github.com/eerwitt/518b0c9564e500b4b50f
we also follow this tutorial https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py


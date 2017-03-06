import os

path = 'dataset/train'
all_label = []

f = open("dataset/label", "w")

for fl in os.listdir(path):
    if 'cat' in fl:
        label = '0'
    elif 'dog' in fl:
        label = '1'
    else:
        label = '2'

    print fl

    f.write(label)
    f.write('\n')

f.close()
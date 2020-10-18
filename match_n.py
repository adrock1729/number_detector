import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
import tensorflow as tf
from tqdm import tqdm
from time import time

def match_n(img, num):
    templates = load_n_template(num)
    res_locs = []
    for template in templates:
        res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        res_locs.append(np.array(res))
    return res_locs

def load_n_template(n):
    template_list_binary = sub.run(["ls", str(n)], stdout=sub.PIPE).stdout.decode("utf-8")
    template_list = [str(n)+"/"+x for x in template_list_binary.split("\n") if x != ""]
    templates = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_RGB2BGR) for x in template_list]
    return templates

def calc_match_size(res_loc, th):
    res = np.where(res_loc > th)
    return res[0].shape[0]

def load_template_dataset():
    all_templates = []
    all_label = []
    for num in range(10):
        templates = load_n_template(num)
        labeled_templates_num = [num for x in templates]
        all_templates += templates
        all_label += labeled_templates_num
    return all_templates, all_label

if __name__=="__main__":
    all_data, labels = load_template_dataset()
    all_data = np.array(all_data)
    labels = np.array(labels)
    all_data = all_data / 255.0
    from tensorflow.keras import layers, models
    model = models.Sequential()
    model.add(layers.Conv2D(39, (3, 3), activation='relu', input_shape=(36, 26, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(78, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(78, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(78, activation='relu'))
    model.add(layers.Dense(15, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    test_result = []
    for _ in range(12):
        train_img_indices = np.random.randint(0,len(labels),300)
        train_data = all_data[train_img_indices]
        train_labels = labels[train_img_indices]
        test_img_indices = np.array([x for x in range(len(labels)) if x not in train_img_indices])
        test_data = all_data[test_img_indices]
        test_labels = labels[test_img_indices]
        model.fit(train_data, train_labels, epochs=3)
        test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
        test_result.append(test_acc)
    print(test_result)
    for n in range(10):
        filename = sub.run(["ls",str(n)], stdout=sub.PIPE).stdout.decode("utf-8").split("\n")[50]
        img = cv2.imread(str(n)+"/"+filename)
        img_cvtd = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        start = time()
        predict = np.argmax(model.predict(np.array([img/255.0]))[0])
        end = time()
        print(predict)
        print("predict time: ", (end - start)/1000)

    # plt.plot(test_result)
    # plt.show()

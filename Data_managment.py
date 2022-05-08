import csv
import random
import os


# loads data and returns it in 2 lists- data and labels
def load_raw_data(path, limit = 1000000):

    train_raw = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for line in csv_reader:
            if line:
                i +=1
                train_raw.append(line)
                if i >= limit:
                    break

    train_targets = []
    for each in train_raw:
        if each:
            label = each.pop(0)
            train_targets.append(label)
    return train_raw, train_targets


# returns normalized data
def normalize_raw_data(raw):
    new_data = []
    for each in raw:
        if each:
            new_data.append([int(x)/255 for x in each])
    return new_data


# returns data as list of 10 elements
def normalize_target_data(raw):
    new_data = []
    for each in raw:
        if each:
            normalized = [0 for x in range(10)]
            normalized[int(each)] = 1
            new_data.append(normalized)
    return new_data


def save_normalized_train_data(normal_train_data, new_file_name):
    with open(new_file_name, 'w', newline='') as newfile:
        csv_writer = csv.writer(newfile)
        for each in normal_train_data:
            if each:
                csv_writer.writerow(each)


def save_normalized_labels_data(normal_train_labels, new_file_name):
    with open(new_file_name, 'w', newline='') as newfile:
        csv_writer = csv.writer(newfile)
        for each in normal_train_labels:
            if each:
                csv_writer.writerow(each)


# downloads data, shuffles it, saves it again
def shuffle(data_file, labels_file):
    data = []
    labels = []
    seed = random.random()
    with open(data_file, 'r') as file:
        csv_reader = csv.reader(file)
        i = 0
        for line in csv_reader:
            data.append(line)
            i += 1
            if i >= 60000:
                break

    print('downloaded...', end=' ')
    random.seed(seed)
    random.shuffle(data)
    os.remove(data_file)

    print('shuffled...', end=' ')
    with open(data_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in data:
            writer.writerow(line)
    data = []
    print('saved')

    with open(labels_file, 'r') as file:
        csv_reader = csv.reader(file)
        i = 0
        for line in csv_reader:
            labels.append(line)
            i += 1
            if i >= 60000:
                break
    print('downloaded...', end=' ')
    random.seed(seed)
    random.shuffle(labels)
    os.remove(labels_file)
    print('shuffled...', end=' ')

    with open(labels_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in labels:
            writer.writerow(line)
    labels = []
    print('saved')

# script to save normalized data

'''
if __name__ == "__main__":


    raw_train_data, raw_train_labels = load_raw_data('mnist_train.csv')

    train_data = normalize_raw_data(raw_train_data)
    train_labels = normalize_target_data(raw_train_labels)
    save_normalized_train_data(train_data, 'training_data.csv')
    save_normalized_labels_data(train_labels, 'training_labels.csv')

raw_test_data, raw_test_labels = load_raw_data('mnist_test.csv')

test_data = normalize_raw_data(raw_test_data)
test_labels = normalize_target_data(raw_test_labels)

save_normalized_train_data(test_data, 'test_data.csv')
save_normalized_labels_data(test_labels, 'test_labels.csv')



data, labels = load_raw_data('mnist_train.csv', 1000)
normal_data = normalize_raw_data(data)
normal_labels = normalize_target_data(labels)
save_normalized_train_data(normal_data, 'test_data.csv')
save_normalized_labels_data(normal_labels, 'test_labels.csv')
'''

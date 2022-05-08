import csv
import NN  # my neural network library
import numpy as np
import os
import time
from Data_managment import shuffle


# store training data as generator
def get_training_data(path):
    training_data = []
    with open(path, 'r') as training_data_csv:
        reader_file = csv.reader(training_data_csv)
        for line in reader_file:
            if line:
                training_data.append([float(x) for x in line])
    return (x for x in training_data)


# store training labels as generator
def get_training_labels(path):
    labels_data = []
    with open(path, 'r') as labels_data_csv:
        reader_file = csv.reader(labels_data_csv)
        for line in reader_file:
            if line:
                labels_data.append([int(x) for x in line])
    return (x for x in labels_data)


# store test data as np array
def get_test_data(path):
    training_data = []
    with open(path, 'r') as training_data_csv:
        reader_file = csv.reader(training_data_csv)
        i = 0
        for line in reader_file:
            if line:
                i += 1
                training_data.append([float(x) for x in line])
            
    return np.array([x for x in training_data])


# store test labels as np array
def get_test_labels(path):
    labels_data = []
    with open(path, 'r') as labels_data_csv:
        reader_file = csv.reader(labels_data_csv)
        i = 0
        for line in reader_file:
            if line:
                i += 1
                labels_data.append([int(x) for x in line])
            
    return np.array([x for x in labels_data])


# returns percentage of right answers
def validate_epoch(nn, test_data, test_labels):
    right = 0
    total = 0

    for x, target in zip(test_data, test_labels):
        x = nn.guess(x)
        x = x.reshape(1, 10)
        x = x.tolist()[0]
        if type(target) != list:
            target = target.tolist()
        if x.index(max(x)) == target.index(max(target)):
            right += 1
        total += 1
    return round(right/total*100, 3)


def train_and_test(nn, epochs, testing_data, testing_labels, saving=False):
    validation_start = time.time()
    global correctness
    correctness = validate_epoch(nn, testing_data, testing_labels)
    validation_end = time.time()

    print(f'Epoch 0: {correctness}%, (testing: {round(validation_end-validation_start, 1)}s)', end = '')
    print('...')
    training_data = []
    training_labels = []

    start = time.time()
    shuffle('training_data.csv', 'training_labels.csv')
    training_data = get_training_data('training_data.csv')
    training_labels = get_training_labels('training_labels.csv')
    end = time.time()

    for i in range(epochs): 
        training_start = time.time()
        nn.train(training_data, training_labels)
        training_end = time.time()
        validation_start = time.time()
        correctness = validate_epoch(nn, testing_data, testing_labels)
        validation_end = time.time()
        print(f'Epoch {i+1}: {correctness}% (loading: {round(end-start, 1)}s, '
            f'training: {round(training_end-training_start, 1)}s, '
            f'testing: {round(validation_end-validation_start, 1)}s)')
        if saving:
            save_network(nn, correctness)



# save network's weights and biases as .npy file
def save_network(nn, correctness):
    dictionary = {'weights': np.array(nn.weights), 'biases': np.array(nn.biases)}
    path = 'C:/Users/HP/PycharmProjects/Neural_network_lib/Networks_database/' + str(correctness)
    np.save(path, dictionary)


# assign weights and biases to network
def load_newtork(nn, correctness):
    path = 'C:/Users/HP/PycharmProjects/Neural_network_lib/Networks_database/' + str(correctness) + '.npy'
    dictionary = np.load(path, allow_pickle=True)
    weights = dictionary.item().get('weights')
    biases = dictionary.item().get('biases')
    nn.weights = weights
    nn.biases = biases


# deletes all but five best networks in Networks_database
def leave_top_five():
    directory = os.listdir('C:/Users/HP/PycharmProjects/Neural_network_lib/Networks_database/')
    directory = sorted([float(x[:-4]) for x in directory])[:-5]
    for x in directory:
        os.remove('C:/Users/HP/PycharmProjects/Neural_network_lib/Networks_database/'+str(x)+'.npy')






#for x in test_data[0].tolist():
    

if __name__ == "__main__":
    nn = NN.NeuralNetwork([784, 30, 10])
    #load_newtork(nn, 73.87)  # 70.7
    nn.learning_rate = 0.75
    start = time.time()
    print('ACCESSING TESTING DATA...')
    test_data = get_test_data('test_data.csv')
    test_labels = get_test_labels('test_labels.csv')
    end = time.time()
    print(f'TESTING DATA STORED IN {round(end-start, 1)}s\n')
    load_newtork(nn, 93.1)   #
    print(nn.guess(test_data[0]))
    print(test_labels[0])
    # nn.learning_rate = 1
    train_and_test(nn, 1, test_data, test_labels, saving=True)


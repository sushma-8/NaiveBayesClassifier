"""
Naive bayes test code
"""
import re
import os
import math
import json
import sys

#test_input_path = sys.argv[1]
test_input_path = '/Users/sushma/Documents/CSCI544/CA2/op_spam_training_data_test/'

STOP_NLTK_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                   'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                   'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                   'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                   'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                   'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                   'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                   'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                   'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                   'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                   'now']

IGNORE_WORDS = ['got', 'location', 'front', 'stayed', 'night', 've', 'th', 're', 'get', 'even', 'could', 'us', 'hotel',
                'room', 'rooms', 'chicago', 'would', 'stay', 'one', 'staff', 'service', 'desk', 'made', 'become', 'say', 'mariott',
                'thru','entwistle', 'michigan']

STOP_WORDS = STOP_NLTK_WORDS + IGNORE_WORDS


def NaiveBayesTest():
    train_data = Documents()
    train_data.read_json()
    # Read all '*.txt' files in given input path
    train_data.parse_test_input()


def write_output(fp, out_label, path):
    fp_out = open('nboutput.txt', 'a')
    fp_out.write(out_label.split('_')[1] + ' ' + out_label.split('_')[0] + ' ' + path)
    fp_out.close()


class Documents:

    def __init__(self):
        self.vocabulary = set()
        self.positive_truthful = {}
        self.negative_truthful = {}
        self.positive_deceptive = {}
        self.negative_deceptive = {}
        self.priors = {}
        self.count_dict = {}
        self.class_probability = [self.positive_truthful, self.negative_truthful, self.positive_deceptive,
                                  self.negative_deceptive]
        self.class_names = ['positive_truthful', 'negative_truthful', 'positive_deceptive', 'negative_deceptive']

    def parse_test_input(self):
        fp_out = open('nboutput.txt', 'w')
        fp_out.close()

        for (root, dirs, files) in os.walk(test_input_path, topdown=True):

            for file in files:
                if file.startswith('.') or 'README' in file:
                    continue
                elif file.endswith('.txt'):
                    test_list = []
                    fp = open(root + '/' + file, 'r')
                    temp_line = re.sub(r"[^a-zA-Z']+", ' ', ''.join(fp.readlines()).strip())
                    for word in temp_line.split():
                        word = word.lower()
                        if word in STOP_WORDS:
                            continue
                        test_list.append(word)

                    out_label = self.classify(test_list)
                    write_output(fp, out_label, root + '/' + file + '\n')

    def read_json(self):
        with open('nbmodel.txt', 'r') as json_file:
            temp_list = json.load(json_file)
            self.priors = temp_list[0]
            self.positive_truthful = temp_list[1]
            self.negative_truthful = temp_list[2]
            self.positive_deceptive = temp_list[3]
            self.negative_deceptive = temp_list[4]
            self.count_dict = temp_list[5]
            self.class_probability = [self.positive_truthful, self.negative_truthful, self.positive_deceptive,
                                      self.negative_deceptive]

    def classify(self, test_list):

        output_dict = dict.fromkeys(self.class_names, 0.0)

        for index, class_name in enumerate(self.class_probability):
            output_dict[self.class_names[index]] = self.priors[self.class_names[index]]

            for word in test_list:
                if word in class_name:
                    output_dict[self.class_names[index]] += class_name[word]
                else:
                    output_dict[self.class_names[index]] += math.log2(
                        1 / (self.count_dict[self.class_names[index]] + self.count_dict['vocabulary']))

        return sorted(output_dict, key=output_dict.get, reverse=True)[0]


if __name__ == '__main__':
    NaiveBayesTest()

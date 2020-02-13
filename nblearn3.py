"""
Naive bayes training code
"""
import re
import os
import math
import json
import sys

#train_dir = sys.argv[1]
train_dir = '/Users/sushma/Documents/CSCI544/CA2/op_spam_training_data/'
count = 0
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
                'thru','entwistle', 'i\'m', 'michigan']

STOP_WORDS = STOP_NLTK_WORDS + IGNORE_WORDS



def NaiveBayesTrain():
    train_data = Documents()

    # Read all '*.txt' files in given input path
    train_data.parse_input()

    # Calculate prior probability for each class
    train_data.priors = train_data.calculate_priors()

    # Write model parameters to a file
    train_data.write_model()


class Documents:

    def __init__(self):
        self.vocabulary = set()
        self.positive_truthful = {}
        self.negative_truthful = {}
        self.positive_deceptive = {}
        self.negative_deceptive = {}
        self.priors = {}
        self.class_frequency = [self.positive_truthful, self.negative_truthful, self.positive_deceptive,
                                self.negative_deceptive]
        self.class_names = ['positive_truthful', 'negative_truthful', 'positive_deceptive', 'negative_deceptive']

    def parse_input(self):
        index = -1
        for (root, dirs, files) in os.walk(train_dir, topdown=True):

            if re.search(r".*positive.*truthful.*", root):
                index = 0
            elif re.search(r".*negative.*truthful.*", root):
                index = 1
            elif re.search(r".*positive.*deceptive.*", root):
                index = 2
            elif re.search(r".*negative.*deceptive.*", root):
                index = 3

            if index == -1:
                continue

            for file in files:
                if file.startswith('.'):
                    continue

                fp = open(root + '/' + file, 'r')
                temp_line = re.sub(r"[^a-zA-Z']+", ' ', ''.join(fp.readlines()).strip())
                for word in temp_line.split():
                    word = word.lower()
                    if word in STOP_WORDS:
                        continue
                    if word in self.class_frequency[index]:
                        self.class_frequency[index][word] += 1
                    else:
                        self.class_frequency[index][word] = 1
                    self.vocabulary.add(word)

    def calculate_priors(self):
        # Calculate prior probabilities for each class
        priors = dict()
        total = 0

        for index, class_name in enumerate(self.class_frequency):
            priors[self.class_names[index]] = sum(self.class_frequency[index].values())
            total += len(self.class_frequency[index])

        total_tokens = sum(priors.values())

        for class_name in priors.keys():
            priors[class_name] = math.log2(priors[class_name] / total_tokens)

        return priors

    def write_model(self):

        fp = open('nbmodel.txt', 'w')
        temp_list = [self.priors]
        count_dict = {'vocabulary': len(self.vocabulary)}

        for index, class_freq in enumerate(self.class_frequency):
            output_dict = dict()
            for word in class_freq:
                output_dict[word] = self.calculate_probability(word, class_freq)

            count_dict[self.class_names[index]] = sum(class_freq.values())
            temp_list.append(output_dict)

        temp_list.append(count_dict)
        json.dump(temp_list, fp)
        fp.close()

    def calculate_probability(self, word, label):
        return math.log2((label[word] + 1) / (sum(label.values()) + len(self.vocabulary)))


if __name__ == '__main__':
    NaiveBayesTrain()

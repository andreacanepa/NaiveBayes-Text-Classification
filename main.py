import os
import re
import string
import math
import random
 
DATASET_DIR = 'Dataset'
STOPWORDS_FILE = 'stopwords-en.txt'
NUM_CLASSES = 5

# business      -> 0
# entertainment -> 1
# politics      -> 2
# sport         -> 3
# tech          -> 4

def get_stopwords_list():
    stopwords_list = [line.rstrip('\n') for line in open(STOPWORDS_FILE)]
    return stopwords_list
 
#dataset loading function
def get_dataset(DATASET_DIR):

    data = []
    target = []

    business_files = os.listdir(os.path.join(DATASET_DIR, 'business'))
    for business_file in business_files:
        with open(os.path.join(DATASET_DIR, 'business', business_file), encoding="latin-1") as f:
            data.append(f.read())
            target.append(0) 

    entertainment_files = os.listdir(os.path.join(DATASET_DIR, 'entertainment'))
    for entertainment_file in entertainment_files:
        with open(os.path.join(DATASET_DIR, 'entertainment', entertainment_file), encoding="latin-1") as f:
            data.append(f.read())
            target.append(1)      

    politics_files = os.listdir(os.path.join(DATASET_DIR, 'politics'))
    for politics_file in politics_files:
        with open(os.path.join(DATASET_DIR, 'politics', politics_file), encoding="latin-1") as f:
            data.append(f.read())
            target.append(2)      

    sport_files = os.listdir(os.path.join(DATASET_DIR, 'sport'))
    for sport_file in sport_files:
        with open(os.path.join(DATASET_DIR, 'sport', sport_file), encoding="latin-1") as f:
            data.append(f.read())
            target.append(3)      

    tech_files = os.listdir(os.path.join(DATASET_DIR, 'tech'))
    for tech_file in tech_files:
        with open(os.path.join(DATASET_DIR, 'tech', tech_file), encoding="latin-1") as f:
            data.append(f.read())
            target.append(4)              

    return data, target

class TextClassificator(object):

    #string cleanup removing punctuaction
    def clean_string(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))

    #tokenize strings into words
    def tokenize_string(self, text):
        return re.split("\W+", self.clean_string(text).lower())
 
    #count up how many of each word appears in a list of words
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def train(self, X, Y):
       
        self.num_texts = {}
        self.log_prior_probabilities = {}
        self.bow = {}
        #global vocabulary
        self.global_vocabulary = set() 
        #list containing english stopwords
        self.stopwords_list = get_stopwords_list()

        n = len(X)

        '''compute the LOG class priors by frequency of each category.'''

        self.num_texts['business'] = sum(1 for label in Y if label == 0)
        self.num_texts['entertainment'] = sum(1 for label in Y if label == 1)
        self.num_texts['politics'] = sum(1 for label in Y if label == 2)
        self.num_texts['sport'] = sum(1 for label in Y if label == 3)
        self.num_texts['tech'] = sum(1 for label in Y if label == 4)

        self.log_prior_probabilities['business'] = math.log(self.num_texts['business'] / n)
        self.log_prior_probabilities['entertainment'] = math.log(self.num_texts['entertainment'] / n)
        self.log_prior_probabilities['politics'] = math.log(self.num_texts['politics'] / n)
        self.log_prior_probabilities['sport'] = math.log(self.num_texts['sport'] / n)
        self.log_prior_probabilities['tech'] = math.log(self.num_texts['tech'] / n)

        self.bow['business'] = {}
        self.bow['entertainment'] = {}
        self.bow['politics'] = {}
        self.bow['sport'] = {}
        self.bow['tech'] = {}
    
        #X,Y are iterables
        for x, y in zip(X, Y): 

            if y == 0:
                c = 'business'
            elif y == 1:
                c = 'entertainment'
            elif y == 2:
                c = 'politics'
            elif y == 3:
                c = 'sport'
            else:
                c = 'tech'

            counts = self.get_word_counts(self.tokenize_string(x))
            for word, count in counts.items():
                #removing stop words
                if (word not in self.stopwords_list):
                    if word not in self.global_vocabulary:
                        self.global_vocabulary.add(word)
                    if word not in self.bow[c]:
                        self.bow[c][word] = 0.0
    
                    self.bow[c][word] += count

    def predict(self, X):

        result = []
        for x in X:

            counts = self.get_word_counts(self.tokenize_string(x))

            business_score = 0
            entertainment_score = 0
            politics_score = 0
            sport_score = 0
            tech_score = 0

            for word, _ in counts.items():
                if word not in self.global_vocabulary: continue
                
                # get returns 0.0 if the value is not found -> add Laplace smoothing
                log_w_given_business = math.log( (self.bow['business'].get(word, 0.0) + 1) / (self.num_texts['business'] + len(self.global_vocabulary)) )
                log_w_given_entertainment = math.log( (self.bow['entertainment'].get(word, 0.0) + 1) / (self.num_texts['entertainment'] + len(self.global_vocabulary)) )
                log_w_given_politics = math.log( (self.bow['politics'].get(word, 0.0) + 1) / (self.num_texts['politics'] + len(self.global_vocabulary)) )
                log_w_given_sport = math.log( (self.bow['sport'].get(word, 0.0) + 1) / (self.num_texts['sport'] + len(self.global_vocabulary)) )
                log_w_given_tech = math.log( (self.bow['tech'].get(word, 0.0) + 1) / (self.num_texts['tech'] + len(self.global_vocabulary)) )


                business_score += log_w_given_business
                entertainment_score += log_w_given_entertainment
                politics_score += log_w_given_politics
                sport_score += log_w_given_sport
                tech_score += log_w_given_tech
    
            business_score += self.log_prior_probabilities['business']
            entertainment_score += self.log_prior_probabilities['entertainment']
            politics_score += self.log_prior_probabilities['politics']
            sport_score += self.log_prior_probabilities['sport']
            tech_score += self.log_prior_probabilities['tech']

            #compute result
            aux = {0: business_score, 1: entertainment_score, 2: politics_score, 3: sport_score, 4:tech_score }
            maxValue = max(aux.values())

            for k,v in aux.items():
                if v == maxValue:
                    result_to_append = k

            result.append(result_to_append)
            
        return result

        
if __name__ == "__main__":

    X, y = get_dataset(DATASET_DIR)

    aux = list(zip(X,y))
    random.shuffle(aux)
    X, y = zip(*aux)

    NBC = TextClassificator()
    NBC.train(X[int(len(X)/NUM_CLASSES):], y[int(len(X)/NUM_CLASSES):])
    
    pred = NBC.predict(X[:int(len(X)/NUM_CLASSES)])
    true = y[:int(len(X)/NUM_CLASSES)]
 
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))

import string
import time
import numpy as np
import pandas as pd

negative_file_adr = '../dataset/rt-polarity.neg'
positive_file_adr = '../dataset/rt-polarity.pos'

# reading dataset
start = time.time()
with open(negative_file_adr, 'r') as neg_file:
    neg_lines = neg_file.readlines()
    
    
with open(positive_file_adr, 'r') as pos_file:
    pos_lines = pos_file.readlines()
end = time.time()

print(f'elapsed time: {end - start}')

print(f'number of lines in neg_lines: {len(neg_lines)}')
print(f'example of line in neg_lines: {neg_lines[0]}')
print(f'number of lines in pos_lines: {len(pos_lines)}')
print(f'example of line in pos_lines: {pos_lines[0]}')


from collections import OrderedDict
import traceback

class LanguageModel:
    def __init__(self, lines, weights, test_dataset_count, online=True):
        self.lines = lines         # List of strings (each string is a line in input file)
        self.test_dataset = None   # List of lists (each list is a list of words in line)
        self.train_dataset = None  # List of lists (each list is a list of words in line)
        self.word_count  = None    # number of words in train_dataset
        self.token_count = None    # number of types
        self.interpolation_weights = weights
        self.test_dataset_count = test_dataset_count
        self.online = online
        
        self.unigram_count_dict = None
        self.bigram_count_dict = None        
        self.interpolation_matrix = None
        
        # preparing train and test dataset
        self._make_train_and_test_dataset()
        
        # making count dictionaries - both of them are ordered dictionary 
        self.unigram_count_dict = self._make_unigram_dictionary(self.train_dataset)
        self.bigram_count_dict = self._make_bigram_dictionary(self.train_dataset)
    
        # initialization word_count and token_count and ckeck that count dictionaries correctly created
        unigram_word_count = sum([value for key, value in self.unigram_count_dict.items()])
        dataset_word_count = sum([len(word_list) for word_list in self.train_dataset])
        assert unigram_word_count == dataset_word_count, 'something in wrong in unigram count dictioary creation'
        self.word_count = dataset_word_count
        self.token_count = len(self.unigram_count_dict)
        
        bigram_word_count = sum([value for key, value in self.bigram_count_dict.items()])
        dataset_double_combination_count = sum([len(word_list) - 1 for word_list in self.train_dataset])
        assert bigram_word_count == dataset_double_combination_count, 'something is wrong in bigram count dictionary creation'
        
        
        # making unigram dataframe
        unigram_probs = []
        for word in self.unigram_count_dict:
            unigram_probs.append(self._smoothed_unigram_probability(word))
            
        unigram_data = {
            'count': list(self.unigram_count_dict.values()),
            'prob': unigram_probs,
        }
        self.unigram_df = pd.DataFrame(unigram_data, index=list(self.unigram_count_dict.keys()))
               
        
        if not online:
            # making bigram dataframe
            bigram_probs = []
            for combination in self.bigram_count_dict:
                bigram_probs.append(self._smoothed_bigram_probability(combination))

            rows = len(bigram_probs)
            ## first column in matrix - bigram probabilities
            bigram_col  = np.array(bigram_probs).reshape(rows, 1)

            ## socond column in matrix - probabilites of second word in combinaition
            unigram_col = []
            for combination in self.bigram_count_dict:
                unigram_col.append(self.get_or_calculate_word_probability(combination[1]))
            unigram_col = np.array(unigram_col).reshape(rows, 1)

            ## third clumn in matrix - epsilon
            epsilon_col = np.array([self.interpolation_weights[3] for _ in range(rows)]).reshape(rows, 1)

            ## making martix
            self.interpolation_matrix = np.hstack((bigram_col, unigram_col, epsilon_col))

            bigram_data = {
                'count': list(self.bigram_count_dict.values()),
                'prob': bigram_probs,
                'inter_prob': self._matrix_interpolation_calculator(self.interpolation_weights)
            }
            self.bigram_df = pd.DataFrame(bigram_data, index=pd.MultiIndex.from_tuples(list(self.bigram_count_dict.keys())))

        
    def _make_train_and_test_dataset(self):
        # selecting some lines for testing
        test_dataset_lines  = np.random.choice(self.lines, self.test_dataset_count, replace=False)
        train_dataset_lines = [n for n in self.lines if n not in test_dataset_lines]
        
        self.test_dataset  = self._clean(test_dataset_lines)
        self.train_dataset = self._clean(train_dataset_lines)
        
    
    # gets list of string (lines) and returns a list of 'list of words'(=line)
    @staticmethod
    def _clean(lines): 
        start_symbol = '<s>'
        end_symbol = '</s>'
        cleaned_up = []

        for line in lines:
            line = line.translate({ord(x): ' ' for x in string.punctuation})
            line = line.translate({ord(x): ' ' for x in line if x not in string.printable})
            line = line.split()
            line.insert(0, start_symbol)
            line.append(end_symbol)
            cleaned_up.append(line)

        return cleaned_up

    def _make_unigram_dictionary(self, dataset):
        dictionary = {}
        for line in dataset:
            for word in line:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1
        return OrderedDict(sorted(dictionary.items()))

    
    def _make_bigram_dictionary(self, dataset):
        dictionary = {}
        for line in dataset:
            for i in range(len(line) - 1):
                first_word  = line[i]
                second_word = line[i+1]
                combination = (first_word, second_word)
                if combination in dictionary:
                    dictionary[combination] += 1
                else:
                    dictionary[combination] = 1

        return OrderedDict(sorted(dictionary.items()))
    
    
    # using Laplace smoothing
    def _smoothed_unigram_probability(self, word):
        count = 0
        try: count = self.unigram_count_dict[word]
        except: pass
        
        return (count + 1) / (self.word_count + self.word_count)
    
    
    # using laplace smoothing
    def _smoothed_bigram_probability(self, combination):        
        first_word  = combination[0]
        second_word = combination[1]
        
        count_combination = 0
        try: count_combination = self.bigram_count_dict[combination]
        except: pass
        
        count_first_word = 0
        try: count_first_word = unigram_count_dict[first_word]
        except: pass
        
        return (count_combination + 1) / (count_first_word + self.token_count)
    
    
    def _interpolated_bigram_probability(self, combination):
        assert self.interpolation_weights is not None, 'please set interpolation weights before calculations'
        l3 = self.interpolation_weights[0]
        l2 = self.interpolation_weights[1]
        l1 = self.interpolation_weights[2]
        e  = self.interpolation_weights[3]
        
        first_word  = combination[0]
        second_word = combination[1]
        
        return l3 * self._smoothed_bigram_probability(combination) + \
               l2 * self._smoothed_unigram_probability(second_word) + \
               l1 * e
    

    def _matrix_interpolation_calculator(self, weights):
        return np.matmul(self.interpolation_matrix, np.array(weights[:3]))
    
    
    def get_or_calculate_word_probability(self, word):
        if word in self.unigram_count_dict and not self.online:
            return self.unigram_df.loc[word].at['prob']
        return self._smoothed_unigram_probability(word)
    
    
    def get_or_calculate_combination_probability(self, combination):
        if combination in self.bigram_count_dict and not self.online:
            return self.bigram_df.loc[combination].at['inter_prob']
        return self._interpolated_bigram_probability(combination)
    
    
    def sentence_probability_bigram(self, sentence, log=False):
        total_probability = self.get_or_calculate_word_probability(sentence[0])
        if log: print(f'w0  : {sentence[0]:<47}, prob:{total_probability:<30}')

        for i in range(1, len(sentence) - 1):
            first_word = sentence[i]
            second_word = sentence[i+1]
            p = self.get_or_calculate_combination_probability((first_word, second_word))
            total_probability *= p

            if log: print(f'w{i:<3}: {first_word:<20}, w{i+1:<3}:{second_word:<20}, prob:{p:<30}')

        if log: print(f'FINAL PROBABILITY: {total_probability:}\n')
        return total_probability
    
    
    def sentence_probability_unigram(self, sentence, log=False):
        total_probability = self.get_or_calculate_word_probability(sentence[0])
        if log: print(f'w0  : {sentence[0]:<20}, prob:{total_probability:<20}')
            
        for i in range(1, len(sentence) - 1):
            word = sentence[i]
            p = self.get_or_calculate_word_probability(word)
            total_probability *= p
            
            if log: print(f'w{i:<3}: {word:<20}, prob:{total_probability:<20}')
                
        if log: print(f'FINAL PROBABILITY: {total_probability:}\n')
        return total_probability

    
    def check_on_dataset(self, test_dataset, log=False, mode='bigram'):
        if mode == 'bigram':
            probabilities = []
            for sentence in test_dataset:
                probabilities.append(self.sentence_probability_bigram(sentence, log))
            return probabilities
        else:
            probabilities = []
            
            for sentence in test_dataset:
                probabilities.append(self.sentence_probability_unigram(sentence, log))
            return probabilities 
    
    def set_interpolation_weights(self, weights, update_interpolations=False):
        self.interpolation_weights = weights
        if not self.online and update_interpolations:
            self.bigram_df['inter_prob'] = self._matrix_interpolation_calculator(weights)


def accuracy_test(language_model0, language_model1, log=False, mode='bigram'):

    successful_detection = 0
    # for testing the accuracy of language model 0
    # see how confident it is for detecting its own sentences 
    # if language 1 more confidentely says the sentence belongs to itself, this is a incorrect detection for language model 0
    start = time.time()
    lang0_probs_on_test_dataset0 = language_model0.check_on_dataset(language_model0.test_dataset, log=log, mode=mode)
    lang1_probs_on_test_dataset0 = language_model1.check_on_dataset(language_model0.test_dataset, log=log, mode=mode)
    
    # for testing the accuracy of language model 1
    # see how confident it is for detecting tis own sentences 
    # if language 0 more confidentely says the sentence belongs to itself, this is a incorrect detection for language model 1
    lang0_probs_on_test_dataset1 = language_model0.check_on_dataset(language_model1.test_dataset, log=log, mode=mode)
    lang1_probs_on_test_dataset1 = language_model1.check_on_dataset(language_model1.test_dataset, log=log, mode=mode)
    end = time.time()
    
#     print(end - start)
    
    assert len(lang0_probs_on_test_dataset0) == len(lang1_probs_on_test_dataset0), 'something is wrong in testing'
    assert len(lang0_probs_on_test_dataset1) == len(lang1_probs_on_test_dataset1), 'something is wrong in testing'
    testcase_count_in_test_dataset0 = len(lang0_probs_on_test_dataset0)
    testcase_count_in_test_dataset1 = len(lang0_probs_on_test_dataset1)

    
    # see how confident language model 0 is for detecting its own sentences 
    correct_detection_by_language_model0 = 0
    for i in range(testcase_count_in_test_dataset0):
        if lang0_probs_on_test_dataset0[i] >= lang1_probs_on_test_dataset0[i]:
            # language model 0 is more confident to detecting its sentences than language model 1
            correct_detection_by_language_model0 += 1
            
            
    # see how confident language model 1 is for detecting its own sentences 
    correct_detection_by_language_model1 = 0
    for i in range(testcase_count_in_test_dataset1):
        if lang0_probs_on_test_dataset1[i] <= lang1_probs_on_test_dataset1[i]:
            # language model 1 is more confident to detecting its sentences than language model 0
            correct_detection_by_language_model1 += 1

    language_model0_accuracy = correct_detection_by_language_model0 / testcase_count_in_test_dataset0
    language_model1_accuracy = correct_detection_by_language_model1 / testcase_count_in_test_dataset1
    
    return language_model0_accuracy, language_model1_accuracy


start = time.time()
weights = np.array([0.09, 0.9, 0.01, 0.01])
language_model0_offline = LanguageModel(neg_lines, weights, 500, online=False)
language_model1_offline = LanguageModel(pos_lines, weights, 500, online=False)
end = time.time()
print(f'elapsed time : {end - start}')


start = time.time()
model0_ac, model1_ac = accuracy_test(language_model0_offline, language_model1_offline, mode='bigram', log=False)
end = time.time()
print(f'elapsed time : {end - start}')
print(f'language model 0 accuracy = {np.round(model0_ac * 100, 3)}')
print(f'language model 1 accuracy = {np.round(model1_ac * 100, 3)}')


start = time.time()
model0_ac, model1_ac = accuracy_test(language_model0_offline, language_model1_offline, mode='unigram', log=False)
end = time.time()
print(f'elapsed time : {end - start}')
print(f'language model 0 accuracy = {np.round(model0_ac * 100, 3)}')
print(f'language model 1 accuracy = {np.round(model1_ac * 100, 3)}')


start = time.time()
weights = np.array([0.09, 0.9, 0.01, 0.01])
language_model0_online = LanguageModel(neg_lines, weights, 500, online=True)
language_model1_online = LanguageModel(pos_lines, weights, 500, online=True)
end = time.time()
print(f'elapsed time : {end - start}')


start = time.time()
model0_ac, model1_ac = accuracy_test(language_model0_online, language_model1_online, mode='bigram', log=False)
end = time.time()
print(f'elapsed time : {end - start}')
print(f'language model 0 accuracy = {np.round(model0_ac * 100, 3)}')
print(f'language model 1 accuracy = {np.round(model1_ac * 100, 3)}')


start = time.time()
model0_ac, model1_ac = accuracy_test(language_model0_online, language_model1_online, mode='unigram', log=False)
end = time.time()
print(f'elapsed time : {end - start}')
print(f'language model 0 accuracy = {np.round(model0_ac * 100, 3)}')
print(f'language model 1 accuracy = {np.round(model1_ac * 100, 3)}')


LanguageModel._clean(['this is some test text'])


print("Bigram probability")
print("Language Model 0")
language_model0_online.sentence_probability_bigram(LanguageModel._clean(['this is some test text'])[0], log=True)
print("Language Model 1")
language_model1_online.sentence_probability_bigram(LanguageModel._clean(['this is some test text'])[0], log=True)

print("Unigram probability")
print("Language Model 0")
language_model0_online.sentence_probability_unigram(LanguageModel._clean(['this is some test text'])[0], log=True)
print("Language Model 1")
language_model1_online.sentence_probability_unigram(LanguageModel._clean(['this is some test text'])[0], log=True)


test_case_count = 500
test_dictionary_relaxed = {}
for i in range(test_case_count):
    scalars = np.random.random(2)
    scalars = np.append(scalars, 0.001) # l1
    scalars = scalars / sum(scalars)
    scalars = np.append(scalars, 0.001) # e
    test_dictionary_relaxed[(i, tuple(scalars))] = None
#     print(f'testcase: {i:<4},{scalars}, SUM[:3]: {sum(scalars[:3])}')

counter = 0
total_time = 0
start = time.time()
for key in test_dictionary_relaxed:
    language_model0.set_interpolation_weights(key[1], update_interpolations=False)
    language_model1.set_interpolation_weights(key[1], update_interpolations=False)
    pos_ac, neg_ac = accuracy_test(language_model0, language_model1, mode='unigram')
    test_dictionary_relaxed[key] = (pos_ac, neg_ac)
    counter += 1
    if counter % (test_case_count / 10) == 0:
        now = time.time()
        print(f'step: {counter}, time: {now - start}')
        total_time += now
        start = time.time()       


import datetime
date = datetime.datetime.now().strftime("get_ipython().run_line_magic("b-%d-%Y-%H-%M-%S")", "")
output_file_adr = f'./results_{test_case_count}_{date}.txt'
with open(output_file_adr, 'w') as result_file_relaxed:
    c = {k: v for k, v in sorted(test_dictionary_relaxed.items(), key=lambda item: sum(item[1]), reverse=True)}
    for key, v in c.items():
        k = list(key[1])
        p = 4
        line = f'testcase: {key[0]:<4}[l3: {np.round(k[0], 4):<6}][l2: {np.round(k[1], 4):<6}][l1: {np.round(k[2], 4):<6}][e: {np.round(k[3], 4):<6}] ==> [positive accuracy: {np.round(v[0], 4):<6}][negative accuracy: {np.round(v[1], 4):<6}]'
        print(line)
        result_file_relaxed.write(line + '\n')

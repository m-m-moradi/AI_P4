import string
import time
import numpy as np

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

neg_test_dataset = np.random.choice(neg_lines, int(np.floor(0.1 * len(neg_lines))), replace=False)
pos_test_dataset = np.random.choice(pos_lines, int(np.floor(0.1 * len(pos_lines))), replace=False)

neg_lines = [n for n in neg_lines if n not in neg_test_dataset]
pos_lines = [n for n in pos_lines if n not in pos_test_dataset]

print(f'number of lines in neg_lines: {len(neg_lines)}')
print(f'example of line in neg_lines: {neg_lines[0]}')
print(f'number of lines in pos_lines: {len(pos_lines)}')
print(f'example of line in pos_lines: {pos_lines[0]}')

print(len(neg_test_dataset))
print(len(pos_test_dataset))



def clean(lines):
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

print(f'example of cleaned line: {clean(neg_lines)[0]}')


from collections import OrderedDict
def make_unigram_dictionary(cleaned_lines):
    dictionary = {}
    for line in cleaned_lines:
        for word in line:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    return OrderedDict(sorted(dictionary.items()))

def make_bigram_dictionary(cleaned_lines):
    dictionary = {}
    for line in cleaned_lines:
        for i in range(len(line) - 1):
            first_word  = line[i]
            second_word = line[i+1]
            combination = first_word + ' ' + second_word
            if combination in dictionary:
                dictionary[combination] += 1
            else:
                dictionary[combination] = 1
                
    return OrderedDict(sorted(dictionary.items()))


start = time.time()
neg_unigram_dict = make_unigram_dictionary(clean(neg_lines))
neg_bigram_dict = make_bigram_dictionary(clean(neg_lines))
pos_unigram_dict = make_unigram_dictionary(clean(pos_lines))
pos_bigram_dict = make_bigram_dictionary(clean(pos_lines))
end = time.time()

print(f'elapsed time: {end - start}')

print('\nfirst 10 records in neg_unigram: (sorted by alphabetical order)\n')
counter = 0
for key, value in neg_unigram_dict.items():
    print(f'  {key}: {value}')
    counter += 1
    if counter == 10:
        break

print('\nfirst 10 records in neg_bigram: (sorted by alphabetical order)\n')
counter = 0
for key, value in neg_bigram_dict.items():
    print(f'  {key}: {value}')
    counter += 1
    if counter == 10:
        break


# for testing
s = 0
for key, value in neg_unigram_dict.items():
    s += value
print(s)

t = 0
for line in clean(neg_lines):
    t += len(line)
print(t)

assert t == s, 'something is wrong in making unigram dictionary'


import math
def probablity_of_sentense(sentence, unigram_dict, bigram_dict, number_of_words, log=False):
    
    # without logarithm
    def unigram_probablity(word, unigram_dict, number_of_words):
        return unigram_dict[word] / number_of_words if word in unigram_dict else 0
    
    def bigram_probablity(first_word, second_word, bigram_dict, unigram_dict, number_of_words):
        combination = first_word + ' ' + second_word
        
        count_combination = 0
        try:
            count_combination = bigram_dict[combination]
        except:
            pass
        
        count_first_word = 0
        try: 
            count_first_word = unigram_dict[first_word]
        except:
            pass
        
        return (count_combination + 1) / (count_first_word + number_of_words)
    
    
#      # with logarithm
#     def unigram_probablity(word, unigram_dict, number_of_words):
#         return math.log10(unigram_dict[word] / number_of_words) if word in unigram_dict else 0
    
#     def bigram_probablity(first_word, second_word, bigram_dict, unigram_dict, number_of_words):
#         combination = first_word + ' ' + second_word
        
#         count_combination = 0
#         try:
#             count_combination = bigram_dict[combination]
#         except:
#             pass
        
#         count_first_word = 0
#         try: 
#             count_first_word = unigram_dict[first_word]
#         except:
#             pass
        
#         return math.log10((count_combination + 1) / (count_first_word + number_of_words))

    
    def interpolated_bigram_probablity(first_word, second_word, uigram_dict, bigram_dict, number_of_words):
        l3 = 0.7
        l2 = 0.2
        l1 = 0.1
        e = 0.1
        
        return l3 * bigram_probablity(first_word, second_word, bigram_dict, unigram_dict, number_of_words) + \
               l2 * unigram_probablity(second_word, unigram_dict, number_of_words) + \
               l1 * e
    
    sentence = clean([sentence])[0]

    total_probablity = unigram_probablity(sentence[0], unigram_dict, number_of_words)
    if log: print(f'w0  : {sentence[0]:<47}, prob:{total_probablity:<30}')
    
    for i in range(1, len(sentence) - 1):
        first_word = sentence[i]
        second_word = sentence[i+1]
        p = interpolated_bigram_probablity(first_word, second_word, unigram_dict, bigram_dict, number_of_words)
        total_probablity *= p
        
        if log: print(f'w{i:<3}: {first_word:<20}, w{i+1:<3}:{second_word:<20}, prob:{p:<30}')
        
    if log: print(f'FINAL PROBABILITY: {total_probablity:}\n')
    return total_probablity


def detect_language(sentence, language0, language1, log=False):
    
    if log: print('in Lang0')
    language0_porbability = probablity_of_sentense(sentence, language0['unigram_dict'], language0['bigram_dict'], language0['words_number'], log=log)
    if log: print('in Lang1')
    language1_porbability = probablity_of_sentense(sentence, language1['unigram_dict'], language1['bigram_dict'], language1['words_number'], log=log)

    
#     if language0_porbability == 0 and language1_porbability get_ipython().getoutput("= 0:")
#         return 1
#     elif language0_porbability get_ipython().getoutput("= 0 and language1_porbability == 0:")
#         return 0
#     elif language0_porbability == 0 and language1_porbability == 0:
#         return -1
#     else :
#         if language0_porbability > language1_porbability:
#             return 0
#         else:
#             return 1
            
    if language0_porbability > language1_porbability:
        return 0
    elif language0_porbability < language1_porbability:
        return 1
    else:
        return -1



positive_words_number = 0
for key, value in pos_unigram_dict.items():
    positive_words_number += value

negative_words_number = 0
for key, value in neg_unigram_dict.items():
    negative_words_number += value
    
negative_language_data ={
    'unigram_dict': neg_unigram_dict,
    'bigram_dict': neg_bigram_dict,
    'words_number': negative_words_number,
}

positive_language_data ={
    'unigram_dict': pos_unigram_dict,
    'bigram_dict': pos_bigram_dict,
    'words_number': positive_words_number,
}

log = False
successful_detection = 0
for positive_sentence in pos_test_dataset:
    detected = detect_language(positive_sentence, positive_language_data, negative_language_data, log=log)
    if log: print(f'detected : {detected}')
    if detected == 0:
        successful_detection += 1
        
print(f'Accuracy: {successful_detection / len(pos_test_dataset)}')

successful_detection = 0
for negative_sentence in neg_test_dataset:
    detected = detect_language(negative_sentence, positive_language_data, negative_language_data, log=log)
    if log: print(f'detected : {detected}')
    if detected == 1:
        successful_detection += 1
        
print(f'Accuracy: {successful_detection / len(neg_test_dataset)}')

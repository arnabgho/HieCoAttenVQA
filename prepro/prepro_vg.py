from random import shuffle , seed
import sys
import os.path
import argparse
import numpy as np
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];


# Preprocess the questions to tokenize the text and process the tokens in the questions

def prepro_question(imgs, params):

    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)

        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()
    return imgs

# Build the vocabulary for questions and answers as well as filter out very less occuring words

def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')

    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab


# Use the word to index dictionary to convert the tokens in test into indices
def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

# Get the top num_ans from the answers available in the dataset

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'
    print '\n'.join(map(str,cw[:20]))

    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

# Encode the questions and answers

def encode_question(imgs, params, wtoi):

    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this sequence
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]

    return label_arrays, label_length, question_id



def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi.get(img['ans'], -1) # -1 means wrong answer.

    return ans_arrays


# Check if the answers are contained in the dictionary else remove the bad questions
def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if img['ans'] in atoi:
            new_imgs.append(img)

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs

# get the distinct image paths from the valid images in the dataset
def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    ques_pos_tmp = {}
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        idx = imgtoi.get(img['img_path'])
        img_pos[i] = idx

        if idx-1 not in ques_pos_tmp:
            ques_pos_tmp[idx-1] = []

        ques_pos_tmp[idx-1].append(i+1)

    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N,3), dtype='uint32')
    ques_pos_len = np.zeros(img_N, dtype='uint32')

    for idx, ques_list in ques_pos_tmp.iteritems():
        ques_pos_len[idx] = len(ques_list)
        for j in range(len(ques_list)):
            ques_pos[idx][j] = ques_list[j]
    return unique_img, img_pos, ques_pos, ques_pos_len



def main(params):

    # Steps catered for the Visual Genome Dataset

    # Load the Training Questions and Testing Questions

    # Get the top answers so that a softmax can be built on top of it

    #Filter Questions which are not in the top answers

    # tokenize and preprocess training questions

    # tokenize and preprocess testing questions

    # create the vocabulary for the questions

    # create a 1 indexed vocab translation table

    # create an inverse of the above table

    # encode the questions of train and test data

    # encode the answers so that only ids remain


    # get the split stats


    # train image should be shuffled , then just use the last val_num images as validation

    # train=0 , val = 1 , test =2

    # create output h5 file for training set


    # create output json file




if __name__=="__main__"




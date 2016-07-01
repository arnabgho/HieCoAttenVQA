"""
Preoricess a raw json dataset into hdf5/json files.

Caption: Use NLTK or split function to get tokens.
"""
from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math

#Start adding notes on what each of the functions do


# Tokenizer removes empty spaces and removes end of line sequences
def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):

    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        captions = img['captions']


        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)
        '''
        caption_tokens=[]
        for s in captions:
            if params['token_method'] == 'nltk':
                c_txt = word_tokenize(str(s).lower())
            else:
                c_txt = tokenize(s)
            caption_tokens=caption_tokens+c_txt
        img['processed_caption_tokens']=caption_tokens'''
        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()
    return imgs

def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
        #for w in img['processed_caption_tokens']:
        #    counts[w] = counts.get(w, 0) + 1


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
        final_caption=[]
        for i,caption in enumerate(img['captions']):
            caption_tokens=word_tokenize( str(caption).lower()  )
            caption=[w if counts.get(w,0) > count_thr else 'UNK' for w in caption_tokens]

            final_caption.append(caption)
        img['final_caption']=final_caption
    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt]
        img['final_question'] = question
        final_caption=[]
        for i,caption in enumerate(img['captions']):
            caption_tokens=word_tokenize( str(caption).lower()  )
            caption=[w if w in wtoi else 'UNK' for w in caption_tokens]
            final_caption.append(caption)
        img['final_caption']=final_caption

    return imgs

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

def encode_captions(imgs, params, wtoi):

    max_length = params['caption_max_length']
    N = len(imgs)
    M = params['num_captions']
    label_arrays = np.zeros((N, M , max_length), dtype='uint32')
    label_length = np.zeros( (N,M) , dtype='uint32')
    #question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        #question_id[question_counter] = img['ques_id']
        #question_counter += 1
        for j in range(M):
            label_length[i,j] = min(max_length, len(img['final_caption'][j])) # record the length of this sequence
            for k,w in enumerate(img['final_caption'][j] ):
                if k < max_length:
                    label_arrays[i,j,k] = ( wtoi['UNK'] , wtoi[w] ) [w in wtoi]

    return label_arrays, label_length



def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi.get(img['ans'], -1) # -1 means wrong answer.

    return ans_arrays

def encode_mc_answer(imgs, atoi):
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if img['ans'] in atoi:
            new_imgs.append(img)

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs

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

def include_captions(imgs,captions):
    result_imgs=[]
    for img in imgs:
        img['captions']=captions[img['img_path'].split('/')[1]   ]['captions']
        result_imgs.append(img)

    return result_imgs

def main(params):

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))
    #imgs_train = imgs_train[:5000]
    #imgs_test = imgs_test[:5000]

    captions_train=json.load(open( params['captions_train_json']  , 'r'))
    captions_test=json.load(open( params['captions_test_json'] , 'r'  ))

    imgs_train=include_captions(imgs_train,captions_train)
    imgs_test=include_captions(imgs_test,captions_test)

   # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w:i+1 for i,w in enumerate(top_ans)}
    itoa = {i+1:w for i,w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)

    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)

    cap_train,cap_length_train=encode_captions(imgs_train,params,wtoi)

    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, params, wtoi)

    cap_test,cap_length_test=encode_captions(imgs_test,params,wtoi)
    # get the unique image for train and test
    unique_img_train, img_pos_train, ques_pos_train, ques_pos_len_train = get_unqiue_img(imgs_train)
    unique_img_test, img_pos_test, ques_pos_test, ques_pos_len_test = get_unqiue_img(imgs_test)

    #########################################
    # Load the json and the corresponding captions

    # Get the top k captions out of every unique img

    # Encode the captions to get the index of each word using the dictionary



    #########################################
    # get the answer encoding.
    ans_train = encode_answer(imgs_train, atoi)

    ans_test = encode_answer(imgs_test, atoi)
    MC_ans_test = encode_mc_answer(imgs_test, atoi)

    # get the split
    N_train = len(imgs_train)
    N_test = len(imgs_test)


    print("N_train")
    print(N_train)

    print("N_test")
    print(N_test)
    # since the train image is already suffled, we just use the last val_num image as validation
    # train = 0, val = 1, test = 2
    split_train = np.zeros(N_train)
    #split_train[N_train - params['val_num']: N_train] = 1

    split_test = np.zeros(N_test)
    split_test[:] = 2

    # create output h5 file for training set.
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)

    ################################
    # For the Captions
    f.create_dataset("cap_train", dtype='uint32', data=cap_train)
    f.create_dataset("cap_test", dtype='uint32', data=cap_test)

    f.create_dataset("cap_len_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("cap_len_test", dtype='uint32', data=ques_length_test)


    ################################

    f.create_dataset("answers", dtype='uint32', data=ans_train)
    f.create_dataset("ans_test", dtype='uint32', data=ans_test)

    f.create_dataset("ques_id_train", dtype='uint32', data=question_id_train)
    f.create_dataset("ques_id_test", dtype='uint32', data=question_id_test)

    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)


    f.create_dataset("ques_pos_train", dtype='uint32', data=ques_pos_train)
    f.create_dataset("ques_pos_test", dtype='uint32', data=ques_pos_test)

    f.create_dataset("ques_pos_len_train", dtype='uint32', data=ques_pos_len_train)
    f.create_dataset("ques_pos_len_test", dtype='uint32', data=ques_pos_len_test)

    f.create_dataset("split_train", dtype='uint32', data=split_train)
    f.create_dataset("split_test", dtype='uint32', data=split_test)

    f.create_dataset("ques_len_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("ques_len_test", dtype='uint32', data=ques_length_test)
    f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)

    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['uniuqe_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json

    parser.add_argument('--num_captions' , default=20 , help ='The number of captions that is to be used')
    parser.add_argument('--captions_train_json' , default='../data/captions_train.json',help='The json file containing all the captions' )
    parser.add_argument('--captions_test_json' , default='../data/captions_test.json',help='The json file containing all the captions' )
    parser.add_argument('--input_train_json', default='../data/vqa_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='../data/vqa_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--output_json', default='../data/vqa_fact_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/vqa_fact_data_prepro.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=26, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    parser.add_argument('--caption_max_length', default=26, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)

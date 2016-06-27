import json
import os
import argparse

# vg : refers to the visual genome dataset for convenience
def download_vg():
    #os.system('wget https://cs.stanford.edu/people/rak248/VG_100K/images.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/image_data.json.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/region_descriptions.json.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/question_answers.json.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/objects.json.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/attributes.json.zip -P zip/')
    os.system('wget https://visualgenome.org/static/data/dataset/relationships.json.zip -P zip/'  )

    os.system('unzip zip/image_data.json.zip -d annotations/')
    os.system('unzip zip/region_descriptions.json.zip -d annotations/')
    os.system('unzip zip/question_answers.json.zip -d annotations/')
    os.system('unzip zip/objects.json.zip -d annotations/')
    os.system('unzip zip/attributes.json.zip -d annotations/')
    os.system('unzip zip/relationships.json.zip -d annotations/')

def extract_vg():
    os.system('unzip zip/image_data.json.zip -d annotations/')
    os.system('unzip zip/region_descriptions.json.zip -d annotations/')
    os.system('unzip zip/question_answers.json.zip -d annotations/')
    os.system('unzip zip/objects.json.zip -d annotations/')
    os.system('unzip zip/attributes.json.zip -d annotations/')
    os.system('unzip zip/relationships.json.zip -d annotations/')



def main(params):
    if params['download']==1:
        download_vg()
    if params['extract']==1:
        extract_vg()

    '''
    Put the VG data into a single json file, [[Question_id,Image_id,Question,multipleChoice_answer,Answer] .. ]
    '''
    question_answers=json.load(open( 'annotations/question_answers.json'))
    relationships=json.load(open( 'annotations/relationships.zip' ))
    train_qa=[]
    train_rel=[]
    if params['split']==1:
        print "Loading Questions and Answers....."
        for i in range(len(question_answers)):
            qa=question_answers[i]['qas']
            # Considering Questions and Answers with one word answers only
            for i in range(len(qa)):


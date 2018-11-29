import math
import numpy as np
import argparse
import random
from random import shuffle
import logging
from datetime import datetime
import json
import sys
import csv
from nltk.tokenize import word_tokenize
from embeddings import *
from WordRNN import *
from collections import Counter
import operator

import time


with open("config.json",'r') as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Autonomous Driving for GTA 5")

parser.add_argument('--num_epochs', default=config['num_epochs'], type=int,
    help='number of training epochs')

parser.add_argument('--batch_size', type=int, default=config['batch_size'],
    help='Batch Size')

parser.add_argument('--gpu', action='store_true', default=config['gpu'],
    help='Use GPU')

parser.add_argument('--gpu_number', default=config['gpu_number'], type=int,
    help='Which GPU to run on')

parser.add_argument('--lr', type=float, default=config['lr'],
    help='Learning rate')

parser.add_argument('--train_ratio', type=float, default=config['train_ratio'],
    help='Ratio for Training Examples')

parser.add_argument('--embedding_path', default=config['embedding_path'], type=str,
    help='path to embeddings')

parser.add_argument('--train_data', default=config['train_data'], type=str,
    help='path to train data')

parser.add_argument('--save_model', default=config['save_model'], type=str,
    help='path to directory to save model weights')

parser.add_argument('--log_dir', default=config['log_dir'], type=str,
    help='path to directory to save logs')

parser.add_argument('--log_name', default=config['log_name'], type=str,
    help='name of the log file starting string')

parser.add_argument('--print_after', default=config['print_after'], type=int,
    help='Print Loss after every n iterations')

parser.add_argument('--validate_after', default=config['validate_after'], type=int,
    help='Validate after every n iterations')

parser.add_argument('--save_after', default=config['save_after'], type=int,
    help='Save after every n iterations')

parser.add_argument('--seed', default=config['seed'], type=int,
    help='Random Seed to Set')

parser.add_argument('--print', action='store_true', default=config['print'],
    help='Print Log Output to stdout')



parser.add_argument('--word_cell_size', type=int, default=config['word_cell_size'],
    help='Cell cize for word RNN model')

parser.add_argument('--word_num_layers', type=int, default=config['word_num_layers'],
    help='Number of RNN layers in Word level model')

parser.add_argument('--word_sentence_length', type=int, default=config['word_sentence_length'],
    help='Max length of sentence for word level model')

parser.add_argument('--word_cell_type', type=str, default=config['word_cell_type'],
    help='Type of cell (LSTM/GRU) for word level RNN')

parser.add_argument('--vocab_size', type=int, default=config['vocab_size'],
    help='Max length of sentence for word level model')


args = parser.parse_args()

random.seed(args.seed)


logging.basicConfig(level=logging.INFO,
    filename= args.log_dir + args.log_name + datetime.now().strftime('%d_%m_%Y_%H_%M_%S.log'),
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

if args.print:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)


logging.info(args)

def find_k_most_frequent_words(data, k):
    ctr = Counter([word for sublist in data for word in sublist])
    sorted_ctr = sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in sorted_ctr[0:k]]


def load(input_path):
    kaggle_toxic_train_file = input_path + 'ToxicDataKaggle/kaggle_toxic_train.csv'
    kaggle_regular_train_file = input_path + 'ToxicDataKaggle/kaggle_regular_train.csv'
    reddit_sarcasm_train_file = input_path + 'SarcasmRedditData/reddit_sarcasm_train.csv'

    kaggle_toxic_val_file = input_path + 'ToxicDataKaggle/kaggle_toxic_validate.csv'
    kaggle_regular_val_file = input_path + 'ToxicDataKaggle/kaggle_regular_validate.csv'
    reddit_sarcasm_val_file = input_path + 'SarcasmRedditData/reddit_sarcasm_validate.csv'



    toxic_data_x_train, toxic_data_x_len_train, toxic_data_y_train = load_data(kaggle_toxic_train_file, 0)
    sarcasm_data_x_train, sarcasm_data_x_len_train, sarcasm_data_y_train = load_data(reddit_sarcasm_train_file, 1)
    regular_data_x_train, regular_data_x_len_train, regular_data_y_train = load_data(kaggle_regular_train_file, 2)

    #toxic_data_x_val, toxic_data_x_len_val, toxic_data_y_val = load_data(kaggle_toxic_val_file, 0)
    #sarcasm_data_x_val, sarcasm_data_x_len_val, sarcasm_data_y_val = load_data(reddit_sarcasm_val_file, 1)
    #regular_data_x_val, regular_data_x_len_val, regular_data_y_val = load_data(kaggle_regular_val_file, 2)

    data_x_train = toxic_data_x_train + sarcasm_data_x_train + regular_data_x_train
    data_x_len_train = toxic_data_x_len_train + sarcasm_data_x_len_train + regular_data_x_len_train
    data_y_train = toxic_data_y_train + sarcasm_data_y_train + regular_data_y_train

    #data_x_val = toxic_data_x_val + sarcasm_data_x_val + regular_data_x_val
    #data_x_len_val = toxic_data_x_len_val + sarcasm_data_x_len_val + regular_data_x_len_val
    #data_y_val = toxic_data_y_val + sarcasm_data_y_val + regular_data_y_val

    data_x_val = []
    data_x_len_val = []
    data_y_val = []

    combined = list(zip(data_x_train, data_x_len_train, data_y_train))
    random.shuffle(combined)
    data_x_train, data_x_len_train, data_y_train = zip(*combined)

    return data_x_train, data_x_len_train, data_y_train, data_x_val, data_x_len_val, data_y_val



def load_data(input_file, label):
    data_x = []
    data_x_len = []
    data_y = []

    logging.info("Reading from input file from {}".format(input_file))
    with open (input_file, 'r+', encoding="utf-8") as train_data:

        data_reader = csv.reader(train_data, delimiter=',')
        next(data_reader) #to skip header
        
        lines_read = 0
        num_lines = sum(1 for line in data_reader)

        train_data.seek(0)
        data_reader = csv.reader(train_data, delimiter=',')
        next(data_reader) #to skip header
        
        for row in data_reader:
            lines_read += 1
            # if lines_read > 2000:
            #         break

            print("Read {}/{} data points from input".format(lines_read, num_lines), end = '\r')
            if label == 1: #Sarcasm Data. INclude parent also
                comment_words = word_tokenize(row[3].lower() + row[2].lower())
            else:
                comment_words = word_tokenize(row[2].lower())

            comment_words = comment_words.copy()

            if len(comment_words) < args.word_sentence_length:
                comment_words += [PAD_WORD] * (args.word_sentence_length - len(comment_words))
            elif len(comment_words) > args.word_sentence_length:
                comment_words = comment_words[:args.word_sentence_length]

            comment_words = [START_WORD] + comment_words + [END_WORD]
            data_x.append(comment_words)
            data_x_len.append(min(args.word_sentence_length, len(comment_words)) + 2)
            data_y.append(list(np.eye(3)[label]))



    logging.info("Inputs Loaded Successfully")
    return data_x, data_x_len, data_y


def create_batches(batch_size, data_x, data_x_len, data_y):
    
    num_batches = math.ceil(len(data_x)/batch_size)
    batches_x = np.array_split(data_x, num_batches)
    batches_y = np.array_split(data_y, num_batches)
    batches_x_len = np.array_split(data_x_len, num_batches)
    return batches_x, batches_x_len, batches_y



def convertToTokens(data, embedding):
    return [[embedding.look_up_word(word) for word in sublist] for sublist in data]
    

def train_AI(embedding_path, input_path, model_save_path):
    data_x_train, data_x_len_train, data_y_train, data_x_val, data_x_len_val, data_y_val = load(input_path)
    words = find_k_most_frequent_words(data_x_train, args.vocab_size)

    full_embeddings = Embeddings()
    full_embeddings.create_embeddings_from_file(embedding_path)

    reduced_embeddings = Embeddings()
    reduced_embeddings.create_reduced_embeddings(full_embeddings, words)
    
    logging.info("Full Glove shape : {}".format(full_embeddings.glove.shape))
    logging.info("Reduced Glove shape : {}".format(reduced_embeddings.glove.shape))
    data_x_train = convertToTokens(data_x_train, reduced_embeddings)

    batch_x, batch_x_len, batch_y = create_batches(args.batch_size, data_x_train, data_x_len_train,
                                                           data_y_train)



    train_len = int(args.train_ratio*len(batch_x))

    train_data_x, train_data_x_len, train_data_y = batch_x[:train_len], batch_x_len[:train_len], batch_y[:train_len]
    validation_data_x, validation_data_x_len, validation_data_y = batch_x[train_len:], batch_x_len[train_len:], batch_y[train_len:]


    
    logging.info("Number of Training Examples : {}, Number of batches : {}".format(len(train_data_x) * args.batch_size, len(train_data_x)))
    logging.info("Number of Validation Examples : {}, Number of batches : {}".format(len(validation_data_x) * args.batch_size, len(validation_data_x)))


    # wordRNN = WordRNN_TF(logging, args.lr, args.word_cell_size , args.word_num_layers, reduced_embeddings.glove, args.batch_size, args.word_sentence_length, 
    #         args.word_cell_type, sess=tf.Session(), 
    #         grad_clip=1.0, dropout_probability = 0.5)


    wordRNN = WordRNN_Trainer(logging, args.lr, args.word_cell_size, 
        args.word_num_layers, reduced_embeddings.glove, args.batch_size, args.word_sentence_length, args.word_cell_type, 
        args.gpu, args.gpu_number, 3,dropout_probability = 0.5)


    for epoch in range(1, args.num_epochs + 1):
        epoch_loss = 0
        batch_index = 0
        wordRNN.fit(train_data_x, train_data_x_len, train_data_y, epoch)
        
        if epoch % args.save_after == 0:
            wordRNN.save(model_save_path +"_"+str(epoch))

        if epoch % args.validate_after == 0:
            if len(validation_data_x) > 0:
                logging.info("Testing on Validation Set : ")
                wordRNN.test(validation_data_x, validation_data_x_len, validation_data_y, epoch)
            else:
                logging.info("Valiation Set size is 0. Not Testing")

    logging.info("Testing on Test Set")
    #wordRNN.test(test_data_x, test_data_x_len, test_data_y)


    
    
    #for epoch in range(1, args.num_epochs + 1):
    #     epoch_loss = 0
    #     batch_index = 0
    #     trainer.fit(train_data_x, train_data_y, epoch)





if __name__ == "__main__":

    train_AI(args.embedding_path, args.train_data, args.save_model)


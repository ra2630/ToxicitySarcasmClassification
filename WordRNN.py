import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchnet as tnt

import numpy as np

class WordRNN_TF:
    def __init__(
        self, logging, learning_rate, cell_size , n_layers, glove, batch_size, sentence_size, 
            cell_type , sess, grad_clip=1.0, dropout_probability = 0.5):
        
        self.learning_rate = learning_rate
        self.cell_size = cell_size
        self.cell_type = cell_type
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.glove_embedding = glove
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability
        self.sess = sess
        self.sentence_size = sentence_size
        self.logging = logging
        
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        
    def save(self, filepath):
        self.logging.info("Saving Session to file : {}".format(filepath))
        saver = tf.train.Saver()
        saver.save(self.sess, filepath)
        self.logging.info("Session Saved !")
        
    def load(self, filepath):
        self.logging.info("Loading Session from file ".format(filepath))
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath)
        self.sess.run(tf.tables_initializer())
        self.logging.info("Session restored !")


    def build_graph(self):

        self.add_embedding_layer()
        self.add_encoder_layer()
        self.add_backward_path()


    def add_embedding_layer(self): 
        self.embedding = tf.get_variable("embedding", initializer=self.glove_embedding)
        self.embedding = tf.cast(self.embedding, dtype=tf.float32)
        self.embedding_dimensions = self.glove_embedding.shape[1]
        self.vocabulary_size = self.glove_embedding.shape[0]
        
    def add_encoder_layer(self):
        self.d_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.d_tokens_unstacked = tf.unstack(self.d_tokens, self.sentence_size + 2, 1)
        self.d_lengths = tf.placeholder(tf.int32, shape=[None])
        self.a_labels = tf.placeholder(tf.int32, shape=[None, None])
        
        document_emb = tf.nn.embedding_lookup(self.embedding, self.d_tokens_unstacked)

        if self.cell_type == 'LSTM':
            self.forward_cell = tf.contrib.rnn.LSTMCell(self.cell_size)
            self.backward_cell = tf.contrib.rnn.LSTMCell(self.cell_size)

        elif self.cell_type == 'GRU':
            self.forward_cell = tf.contrib.rnn.GRUCell(self.cell_size)
            self.backward_cell = tf.contrib.rnn.GRUCell(self.cell_size)
        
        self.forward_cell = tf.nn.rnn_cell.DropoutWrapper(
            self.forward_cell, 
            output_keep_prob = self.dropout_probability,
            state_keep_prob = self.dropout_probability
        )
        
        self.backward_cell = tf.nn.rnn_cell.DropoutWrapper(
            self.backward_cell, 
            output_keep_prob=self.dropout_probability,
            state_keep_prob = self.dropout_probability
        )

        self.encoder_outputs, self.encoder_states = tf.nn.bidirectional_dynamic_rnn(
            self.forward_cell,
            self.backward_cell, 
            document_emb,
            time_major = False, #TODO : Double check this value
            sequence_length=self.d_lengths,
            dtype=tf.float32
        )
        
        
        if self.cell_type == 'LSTM':
            self.encoder_states = tf.concat(self.encoder_states, 2)
            self.encoder_states = tf.concat(self.encoder_states, 1)
            self.encoder_states = tf.reshape(self.encoder_states, shape = (-1, self.cell_size * 4))
            
        elif self.cell_type == 'GRU':
            self.encoder_states = tf.concat(self.encoder_states, 1)

        self.encoder_states = tf.cast(self.encoder_states,tf.float32)
        self.answer_tags = tf.layers.dense(inputs=self.encoder_states, units=8)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.answer_tags, labels=self.a_labels, name = 'loss'))

    
    def add_backward_path(self):
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)


    def fit(self, train_data_x, train_data_x_len, train_data_y, epoch):

        batch_index = 0
        loss = 0
        acc_session = tf.Session()
        total_acc = 0
        total_labels = 0

        for batch_x, batch_x_len, batch_y in zip(train_data_x, train_data_x_len, train_data_y):
            batch_index += 1 
            data_x = np.array(batch_x, dtype=np.float32)
            data_x_len = np.array(batch_x_len, dtype=np.float32).reshape(len(batch_x))
            data_y = np.array(batch_y, dtype=np.int32)

            _, loss_value, answer_tags = self.sess.run([self.train_op, self.loss, self.answer_tags], {
                self.d_tokens: data_x,
                self.d_lengths: data_x_len,
                self.a_labels: data_y
            })
            prediction = tf.argmax(answer_tags, 1)
            labels = tf.argmax(data_y, 1)
            equality = tf.equal(prediction, labels)
            accuracy = tf.reduce_sum(tf.cast(equality, tf.int32))
            acc = acc_session.run(accuracy)
            total_acc += acc
            total_labels += len(batch_x)
            self.logging.info("Epoch {} Batch {} Loss {} Accuracy {}/{} ({}%)".format(epoch, batch_index, loss_value, acc, len(batch_x), (acc * 100.0)/len(batch_x)))
            loss += loss_value
            
        loss/=len(train_data_x)
        self.logging.info("Epoch {} Average Loss {} Accuracy {}/{} ({}%)".format(epoch, loss, total_acc, total_labels, (total_acc * 100.0)/total_labels))



    def test(self, test_data_x, test_data_x_len, test_data_y):
        batch_index = 0
        loss = 0

        acc_session = tf.Session()
        total_acc = 0
        total_labels = 0

        for batch_x, batch_x_len, batch_y in zip(test_data_x, test_data_x_len, test_data_y):
            batch_index += 1 
            data_x = np.array(batch_x, dtype=np.float32)
            data_x_len = np.array(batch_x_len, dtype=np.float32).reshape(len(batch_x))
            data_y = np.array(batch_y, dtype=np.int32)
            
            loss_value, answer_tags = self.sess.run([self.loss, self.answer_tags], {
                self.d_tokens: data_x,
                self.d_lengths: data_x_len,
                self.a_labels: data_y
            })

            prediction = tf.argmax(answer_tags, 1)
            labels = tf.argmax(data_y, 1)
            equality = tf.equal(prediction, labels)
            accuracy = tf.reduce_sum(tf.cast(equality, tf.int32))
            acc = acc_session.run(accuracy)
            total_acc += acc
            total_labels += len(batch_x)

            self.logging.info("Batch {} Loss {} Accuracy {}/{} ({}%)".format(batch_index, loss_value, acc, len(batch_x), (acc * 100.0)/len(batch_x)))
            loss += loss_value
        
        loss/=len(test_data_x)
        self.logging.info("Test Loss {} Accuracy {}/{} ({}%)".format(loss, total_acc, total_labels, (total_acc * 100.0)/total_labels))






class WordRNN_Torch(nn.Module):
    def __init__(
        self, cell_size , n_layers, glove, cell_type , batch_size, GPU, gpu_number, output_classes,
            dropout_probability = 0.5):

        super(WordRNN_Torch, self).__init__()
        
        self.cell_size = cell_size
        self.cell_type = cell_type
        self.n_layers = n_layers
        self.glove = glove
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability

        self.GPU = GPU
        self.gpu_number = gpu_number


        self.word_embeddings = nn.Embedding(self.glove.shape[0], self.glove.shape[1])
        self.word_embeddings.load_state_dict({'weight': torch.from_numpy(self.glove)})
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(self.glove.shape[1], self.cell_size,
                            num_layers=self.n_layers, bidirectional=True, batch_first = True, dropout = dropout_probability)

        self.linear = nn.Sequential(
                #nn.BatchNorm1d(4 * self.cell_size),
                nn.ReLU(),
                #nn.Dropout(self.dropout_probability),
                nn.Linear(4 * self.cell_size, 512),

                #nn.BatchNorm1d(512),
                nn.ReLU(),
                #nn.Dropout(self.dropout_probability),
                nn.Linear(512, 256),

                #nn.BatchNorm1d(256),
                nn.ReLU(),
                #nn.Dropout(self.dropout_probability),
                nn.Linear(256, 128),

                #nn.BatchNorm1d(128),
                nn.ReLU(),
                #nn.Dropout(self.dropout_probability),
                nn.Linear(128, output_classes),
            )

    def init_hidden(self):
        if self.GPU:
            return (torch.zeros(2, self.batch_size, self.cell_size).cuda(self.gpu_number),
                torch.zeros(2, self.batch_size, self.cell_size).cuda(self.gpu_number))
        else:
            return (torch.zeros(2, self.batch_size, self.cell_size),
                torch.zeros(2, self.batch_size, self.cell_size))


    def update_batchsize(self, batch_size):
        self.batch_size = batch_size


    def forward(self, sentence):   
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)

        _, (self.hidden_state, self.cell_state) = self.lstm(embeds, self.hidden)
        
        self.hidden_state = self.hidden_state.reshape(self.batch_size, -1)
        self.cell_state = self.cell_state.reshape(self.batch_size, -1)

        self.concateneated_state = torch.cat((self.cell_state, self.hidden_state), dim = -1)

        answer_labels = self.linear(self.concateneated_state)
        
        return answer_labels


class WordRNN_Trainer:
    def __init__(
        self, logging, learning_rate, cell_size , n_layers, glove, batch_size, sentence_size, 
            cell_type, GPU, gpu_number, output_classes, dropout_probability = 0.5):

        self.GPU = torch.cuda.device_count() >= 1 and GPU
        self.model = WordRNN_Torch(cell_size, n_layers, glove, cell_type, batch_size, self.GPU, gpu_number,
                                   output_classes, dropout_probability)

        self.output_classes = output_classes
        self.gpu_number = gpu_number
        self.batch_size = batch_size
        self.logging = logging

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def custom_loss_function(self, prediction, gold):
        mask = torch.Tensor(np.array([[1.0,1.0,1.0], [0.0,1.0,1.0], [1.0,0.0,1.0]]))

        if self.GPU:
            mask = mask.cuda(self.gpu_number)

        loss_value = nn.BCELoss(reduce=False)(nn.Sigmoid()(prediction), gold)

        with torch.no_grad():
            for idx, g in enumerate(gold):
                loss_value[idx] = mask[int(torch.argmax(gold[idx].data))] * loss_value[idx]

        return torch.mean(loss_value);






    def fit(self, train_data_x, train_data_x_len, train_data_y, epoch):

        batch_index = 0
        loss = 0

        self.model.train()
        confMatrix = tnt.meter.ConfusionMeter(self.output_classes)

        for batch_x, batch_y in zip(train_data_x, train_data_y):
            batch_index += 1 
            data_x = torch.from_numpy(np.array(batch_x, dtype=np.long))
            data_y = torch.from_numpy(np.array(batch_y, dtype=np.float32))
            #data_y = torch.max(Variable(data_y), 1)[1]

            self.model.update_batchsize(data_x.shape[0])

            if self.GPU:
                data_x = data_x.cuda(self.gpu_number)
                data_y = data_y.cuda(self.gpu_number)
                self.model = self.model.cuda(self.gpu_number)

            self.optimizer.zero_grad()

            prediction = self.model(data_x)

            confMatrix.add(prediction.clone().detach(),data_y.clone().detach())

            loss_value = self.custom_loss_function(prediction, data_y)
            loss_value.backward()
            self.optimizer.step()

            self.logging.info("Epoch {} Batch {} Loss {}".format(epoch, batch_index, loss_value))
            loss += loss_value

        self.logging.info('\nConfusion Matrix on Train for epoch {} \n {}\n'.format(epoch,
                    confMatrix.value()))
            
        loss/=len(train_data_x)
        self.logging.info("Epoch {} Average Loss {}".format(epoch, loss))



    def test(self, test_data_x, test_data_x_len, test_data_y, epoch):

        batch_index = 0
        loss = 0

        confMatrix = tnt.meter.ConfusionMeter(self.output_classes)

        self.model.eval()

        for batch_x, batch_y in zip(test_data_x, test_data_y):
            batch_index += 1 
            data_x = torch.from_numpy(np.array(batch_x, dtype=np.long))
            data_y = torch.from_numpy(np.array(batch_y, dtype=np.float32))
            #data_y = torch.max(Variable(data_y), 1)[1]

            self.model.update_batchsize(data_x.shape[0])

            if self.GPU:
                data_x = data_x.cuda(self.gpu_number)
                data_y = data_y.cuda(self.gpu_number)
                self.model = self.model.cuda(self.gpu_number)


            prediction = self.model(data_x)

            confMatrix.add(prediction.clone().detach(),data_y.clone().detach())

            loss_value = self.custom_loss_function(prediction, data_y)

            self.logging.info("Epoch {} Batch {} Loss {}".format(epoch, batch_index, loss_value))
            loss += loss_value

        self.logging.info('\nConfusion Matrix on Test \n{}\n'.format(
            confMatrix.value()))
            
        loss/=len(test_data_x)
        self.logging.info("Epoch {} Average Loss {}".format(epoch, loss))


    def load(self, filepath):
        self.logging.info("Loading model from file {}".format(filepath))
        self.model.load_state_dict(torch.load(filepath, map_location='cpu'))
        if(self.GPU):
            self.model = self.model.cuda(gpu_number)
        self.logging.info("Model loaded successfully !")


    def save(self, filepath):
        self.logging.info("Saving model to file {}".format(filepath))
        if self.GPU:
            torch.save(self.model.cpu().state_dict(), filepath)
            self.model = self.model.cuda(self.gpu_number)
        else:
            torch.save(self.model.state_dict(), filepath)
        self.logging.info("Model saved successfully !")

    
    def infer(self, sentence):
        self.model.eval()
        data_x = sentence
        self.model.update_batchsize(1)
        if self.GPU:
            data_x = data_x.cuda(self.gpu_number)
        prediction = self.model(data_x)
        prediction = torch.max(Variable(prediction), 1)[1]
        prediction = prediction.cpu()
        return prediction




    
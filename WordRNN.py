import tensorflow as tf
import numpy as np

class WordRNN:
    def __init__(
        self, logging, learning_rate, cell_size , n_layers, glove, batch_size, sentence_size, 
            cell_type = 'GRU', sess=tf.Session(), 
                grad_clip=1.0, dropout_probability = 0.5):
        
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
        
    def saveSession(self, filepath):
        self.logging.info("Saving Session to file : {}".format(filepath))
        saver = tf.train.Saver()
        saver.save(self.sess, filepath)
        self.logging.info("Session Saved !")
        
    def loadFromSession(self, filepath):
        self.logging.info("Loading Session from file ".format(filepath))
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath)
        self.sess.run(tf.tables_initializer())
        self.loging.info("Session restored !")


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
            time_major = True,
            sequence_length=self.d_lengths,
            dtype=tf.float32
        )
        
        
        if self.cell_type == 'LSTM':
            self.encoder_states = tf.concat(self.encoder_states, 2)
            self.encoder_states = tf.concat(self.encoder_states, 1)
            self.encoder_states = tf.reshape(self.encoder_states, shape = (-1, self.embedding_dimensions * 4))
            
        elif cell_type == 'GRU':
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
    
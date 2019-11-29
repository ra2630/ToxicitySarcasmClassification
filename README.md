We propose an end-to-end model based on convolution and recurrent neural network natural language models to address the task of classifying text as sarcastic, toxic or regular. We propose an ensemble model based of Convolution Networks, Long Short Term Memory (LSTM) cells coupled with attention layer and Multi Layer Perceptron with features extracted manually to enhance performance. We train our model on the combined toxicity dataset from Kaggle, and Sarcasm dataset from news headlines also published on Kaggle.

Full project report can be found [here](https://drive.google.com/file/d/1DTa8J_ktCo9tnb6zAsZ32zF6FSeEkZcP/view?usp=sharing)

In this paper, we propose a unique model of an ensemble of 5 distinct models. We primarily use 2 broad classes of Deep Learning Networks - Convolution Neural Networks and Recurrent Neural Networks,  and use both Word level embedding and Character level encoding to train these models, giving us a total of 4 networks. We also use a multi-layer perceptron (MLP) Model with statistical features extracted from the sentence in our final ensemble.

For our deep learning networks, We restrict the size of the sentences to 50 words, and length of characters to 300 words, so as to get a uniform vector representation across all our inputs in the data. Any word/character sequence greater than the threshold lengths is simply truncated, and for sentences with lower counts, we pad these with special symbols <PAD\_WORD>/<PAD\_CHAR> depending on the word and character embeddings inside. The truncation/padding are not performed in the MLP model, and the entire sentence is used to create the features. This guarantees that in the final ensemble, there is at-least one model which has seen the entire data as it is. Each word and character sequence is appended with a <START\_WORD>/<START\_CHAR> tag at the beginning and with a <END\_WORD>/<END\_CHAR> tag at the end of the sentence, to indicate start and end of each sequence.

We use <UNK> keyword to represent the unknown words, and since abusive language plays a key role in toxicity detection, we use a special keyword <ABUSIVE> to replace all abusive words with the above mentioned tag. We also remove all the stop words from our training sentences as they don't play a major role in classifying the sentences.
  



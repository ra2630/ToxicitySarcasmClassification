## Abstract
We propose an end-to-end model based on convolution and recurrent neural network natural language models to address the task of classifying text as sarcastic, toxic or regular. We propose an ensemble model based of Convolution Networks, Long Short Term Memory (LSTM) cells coupled with attention layer and Multi Layer Perceptron with features extracted manually to enhance performance. We train our model on the combined toxicity dataset from Kaggle, and Sarcasm dataset from news headlines also published on Kaggle.

Full project report can be found [here](https://drive.google.com/file/d/1DTa8J_ktCo9tnb6zAsZ32zF6FSeEkZcP/view?usp=sharing)

## Model Architecture
In this paper, we propose a unique model of an ensemble of 5 distinct models. We primarily use 2 broad classes of Deep Learning Networks - Convolution Neural Networks and Recurrent Neural Networks,  and use both Word level embedding and Character level encoding to train these models, giving us a total of 4 networks. We also use a multi-layer perceptron (MLP) Model with statistical features extracted from the sentence in our final ensemble.

For our deep learning networks, We restrict the size of the sentences to 50 words, and length of characters to 300 words, so as to get a uniform vector representation across all our inputs in the data. Any word/character sequence greater than the threshold lengths is simply truncated, and for sentences with lower counts, we pad these with special symbols <PAD\_WORD>/<PAD\_CHAR> depending on the word and character embeddings inside. The truncation/padding are not performed in the MLP model, and the entire sentence is used to create the features. This guarantees that in the final ensemble, there is at-least one model which has seen the entire data as it is. Each word and character sequence is appended with a <START\_WORD>/<START\_CHAR> tag at the beginning and with a <END\_WORD>/<END\_CHAR> tag at the end of the sentence, to indicate start and end of each sequence.

We use <UNK> keyword to represent the unknown words, and since abusive language plays a key role in toxicity detection, we use a special keyword <ABUSIVE> to replace all abusive words with the above mentioned tag. We also remove all the stop words from our training sentences as they don't play a major role in classifying the sentences.
  
## Results and Conclusion
The final aim of this project was to do a multi-label classification of the comments into the following categories:  1. Toxic; 2. Severe Toxic; 3. Obscene; 4. Threat; 5. Insult (Direct Insults); 6. Sarcasm (Indirect Insults) 7. Regular. The dataset was collected from two different sources, one from Reddit(Sarcasm/Non-sarcasm) and one from Twitter(Toxic comments dataset). The toughest part of the problem was to understand the comments that were sarcastic yet toxic, in an indirect way. There is sarcasm that is just humorous but then there is sarcasm that is an indirect insult which is toxic. The main aim of the project was to experiment with these two datasets to find such comments. We manually annotated a gold set by handpicking comments that were both toxic and sarcastic for our evaluation. Both our models were able to perform well on the individual datasets. That is, the models were able to classify the Toxic Comment data into the following categories: 1. Toxic; 2. Severe Toxic; 3. Obscene; 4. Threat; 5. Insult (Direct Insults); 6. Regular and Reddit Comment Data into sarcastic and non sarcastic. **However, when we combined the two datasets to carry out the desired experiment, the models overfit and fail to perform on the gold set. One possible explanation of this behaviour is that the models learnt to differentiate between the data sources rather the distribution space. The findings have been consolidated in the tables below.

| Model        | Accuracy - Train           | Accuracy - Test  |
| ------------- |:-------------:| -----:|
| Char CNN      | 81.93% | 79.70% |
| Word CNN      | 86.22%      |   80.28% |
| Char RNN | 91.74%    |    90.57% |
| Word RNN | 94.18%      |    90.57% |
Accuracy Results for Toxic comment multilabel classification

| Model        | Accuracy - Train           | Accuracy - Test  |
| ------------- |:-------------:| -----:|
| Char CNN       | 77.42% | 71.11% |
| Word CNN      | 82.93%      |   77.61% |
| Char RNN | 85.80%      |    78.05% |
| Word RNN | 89.17%      |    82.27% |
Accuracy Results for Sarcastic comment binary classification

| Model        | Accuracy - Train           | Accuracy - Test  |
| ------------- |:-------------:| -----:|
| Char CNN       | 83.42% | 30.71% |
| Word CNN      | 87.43%      |   29.94% |
| Char RNN | 96.80%      |    32.60% |
| Word RNN | 98.17%      |    33.87% |
Accuracy Results for combined data multilabel classification


  



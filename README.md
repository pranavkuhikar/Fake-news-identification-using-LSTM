# Fake-news-identification-using-LSTM
In this project, identification of fake news is carried out by LSTM
The aim of the project is to classify the news into two categories fake and true.
The dataset is obtained from kaggle. link: https://www.kaggle.com/competitions/fake-news/data?select=submit.csv
It has 20800 rows and 5 columns i.e. id,	title, author,	text ,and label.
There are also some missing values in the data. we need to handle it. When a dataset contains missing values we can either fill them or drop them. Here, we have dropped the null values using .dropna() command (Since we cannot fill the sentiments randomly). So, this leaves us with 18285 rows. And the data loss is within the limit, so it won't affect the algorithm.
Now we need to separate the data into dependent and independent variables.
It is time to clean the text. Cleaning here implies lowering all the words in the data to avoid repetition of words. For eg 'tree' and 'Tree' are the same and won't add any extra information. The second part in the cleaning process is to get rid of the stopwords. These are the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information for sentimment analysis.
Now we need to perform one-hot representation. 
In one hot encoding, every word which is part of the given text data is written in the form of vectors. only of 1 and 0 . So one hot vector is a vector whose elements are only 1 and 0, with each one hot vector being unique. This allows the word to be identified uniquely by its one hot vector. No two words will have the same vetor representation.
We  also need to use padding since length of each sentence is different. Padding will essentially add vectors [0] either before the actual set of vectors or aftet the set of vectors. This can be selected by specifying the type of padding i.e. pre-zero or post-zero padding.
At this time, we need to compile the model. Since this is an LSTM model, the parameters need to be specified, which are as:
# embedding_vector_features = 40. 
This essentially means that the max. limit of vectors would be 40
# model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
The input length is set as equal to the sentence length. Sentence length is taken as 20. It means that each sentence will have 20 words.
# model.add(LSTM(100))
Here the number 100 denotes the total number of neurons in the network.
# model.add(Dense(1, activation='sigmoid'))
Activation function is taken as sigmoid, since there are two categories to handle.
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Similarly the loss function will be using binary_crossentropy as there are two classes to be handled.
# Sampling of the data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,batch_size=64)
It is important to note that the epochs, batch_size are hyperparameters. After trying various values the best resuts were obtained when epocs were set to 10, and batch_size as 64.
# Confusion Matrix
The accuracy score from the confusion matrix was found to be 0.9103562551781276 i.e. 91% 
# Classififcation Report
  precision    recall  f1-score   support

           0       0.92      0.92      0.92      3419
           1       0.89      0.90      0.90      2616

    accuracy                           0.91      6035
   macro avg       0.91      0.91      0.91      6035
weighted avg       0.91      0.91      0.91      6035

# The precision score, recall score and f1-score are showing a good sign and are thus suitable for the model building.

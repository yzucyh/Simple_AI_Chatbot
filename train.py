import random
import json
import pickle
import numpy

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

targets = json.loads(open('traintarget.json').read())

words = []
classes = []
doc = []
dummy_let = ['?', ",", "'"]

for target in targets['target']:
    for pattern in target['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        doc.append((words_list, target['tag']))
        if target['tag'] not in classes:
            classes.append(target['tag'])
            
words = [lemmatizer.lemmatize(word) for word in words if words not in dummy_let]
words = sorted(set(words))

classes = sorted(set(classes))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training_data = []

output_clear = [0]*len(classes)

for doc_each in doc:
    carry_out = []
    word_patterns = doc_each[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        carry_out.append(1) if word in word_patterns else carry_out.append(0)
    
    row = list(output_clear)
    row[classes.index(doc_each[1])] = 1
    training_data.append([carry_out, row])
    
random.shuffle(training_data)
training_data = numpy.array(training_data)

train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu',))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


# stored in var #
fit_model = model.fit(numpy.array(train_x), numpy.array(train_y), epochs = 200, batch_size=5, verbose = 1)
model.save('chat_model.h5', fit_model)
print("success")
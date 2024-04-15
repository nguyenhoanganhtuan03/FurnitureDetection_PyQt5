import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Load data từ file intents.json
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Xử lý dữ liệu intents.json
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize từng câu thành các từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Thêm các văn bản và nhãn tương ứng vào danh sách documents
        documents.append((w, intent['tag']))
        # Thêm nhãn vào danh sách classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize và loại bỏ từ stopwords
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# In thông tin về dữ liệu
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

# Lưu dữ liệu đã xử lý vào các file pickle
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Khởi tạo dữ liệu huấn luyện
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Xáo trộn dữ liệu
random.shuffle(training)
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]
print("Training data created")

# Xây dựng mô hình
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile mô hình
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Lưu mô hình
model.save('chatbot_model.h5', hist)

print("Model created")

import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.layers import Merge
from keras.preprocessing import text,sequence
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

train_df = pd.read_csv('train.csv')
y = train_df.entailment_judgment.values

tknzr = text.Tokenizer(num_words=250000)
max_len = 100

tknzr.fit_on_texts(list(train_df.sentence_A.values.astype(str)))
sentence_a = tknzr.texts_to_sequences(train_df.sentence_A.values.astype(str))
sentence_a = sequence.pad_sequences(sentence_a, maxlen=max_len)
sentence_b = tknzr.texts_to_sequences(train_df.sentence_B.values.astype(str))
sentence_b = sequence.pad_sequences(sentence_b, maxlen=max_len)

word_index = tknzr.word_index

ytrain_enc = np_utils.to_categorical(y)

model1 = Sequential()
model1.add(Embedding(len(word_index)+1, 300, input_length=100, dropout=0.3))
model1.add(LSTM(300,dropout=0.3,recurrent_dropout=0.2))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, input_length=100, dropout=0.3))
model2.add(LSTM(300, dropout=0.3, recurrent_dropout=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
merged_model.fit([sentence_a, sentence_b], y=y, batch_size=250, epochs=10,verbose=1)

output_df = pd.read_csv("test.csv")

output_tknzr = text.Tokenizer(num_words=250000)

output_tknzr.fit_on_texts(list(output_df.sentence_A.values.astype(str))+list(output_df.sentence_B.values.astype(str)))
output_a = tknzr.texts_to_sequences(output_df.sentence_A.values.astype(str))
output_a = sequence.pad_sequences(output_a, maxlen=max_len)

output_b = tknzr.texts_to_sequences(output_df.sentence_B.values.astype(str))
output_b = sequence.pad_sequences(output_b, maxlen=max_len)

result = merged_model.predict_proba([output_a,output_b], batch_size=384, verbose=1)

with open('submission.csv', 'w') as submission_file:
    submission_file.write('test_id,is_duplicate' + '\n')

    for i in range(0, len(result)):
        submission_file.write(str(i) + ',' + str(result[i][0]) + '\n')

# serialize model to JSON
model_json = merged_model.to_json()
with open("merged_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
merged_model.save_weights("merged_model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('merged_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("merged_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.predict_proba([output_a,output_b], batch_size=384, verbose=1)
#
# with open('submission2.csv', 'w') as submission_file:
#     submission_file.write('test_id,semantic_similarity' + '\n')
#
#     for i in range(0, len(score)):
#         submission_file.write(str(i) + ',' + str('%.1f' % score[i][0]) + '\n')
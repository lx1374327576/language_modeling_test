from trigram_model import TrigramModel
from ngram_dataset import TrigramDataset
import numpy as np

trainset = TrigramDataset('./data/penn/train.txt')
validset = TrigramDataset('./data/penn/valid.txt')
testset = TrigramDataset('./data/penn/test.txt')
model = TrigramModel()
model.train(trainset)

train_result = model.predict(trainset)
train_result = np.array(train_result)
print('train: ', np.power(2, -np.average(np.log(train_result))))

valid_result = model.predict(validset)
valid_result = np.array(valid_result)
print('valid: ', np.power(2, -np.average(np.log(valid_result))))

test_result = model.predict(testset)
test_result = np.array(test_result)
print('test: ', np.power(2, -np.average(np.log(test_result))))

# word1 = 'edison'
# word2 = 'spokesman'
# sentence = word1 + ' ' + word2
# for i in range(100):
#     word = model.predict_single(word1, word2)
#     sentence += ' ' + word
#     word1 = word2
#     word2 = word
# print(sentence)

from trigram_model import TrigramModel
from ngram_dataset import TrigramDataset
import numpy as np

trainset = TrigramDataset('./data/penn/train.txt')
validset = TrigramDataset('./data/penn/valid.txt')
testset = TrigramDataset('./data/penn/test.txt')
model = TrigramModel()
model.train(trainset)

train_result = model.predict(trainset)
valid_result = model.predict(validset)
test_result = model.predict(testset)

train_result = np.array(train_result)
valid_result = np.array(valid_result)
test_result = np.array(test_result)

print('train: ', np.sum(np.log(train_result)))
print('valid: ', np.sum(np.log(valid_result)))
print('test: ', np.sum(np.log(test_result)))


class TrigramModel:

    def __init__(self):
        self.count = 0
        self.word_dict = {}
        self.train_data = None

    def train(self, data):

        train_dict = {}
        for key in data:
            a, b, c = key
            if a not in train_dict:
                train_dict[a] = {}
            if b not in train_dict[a]:
                train_dict[a][b] = {}
            if self.count not in train_dict[a][b]:
                train_dict[a][b][self.count] = 0
            if c not in train_dict[a][b]:
                train_dict[a][b][c] = 1
                train_dict[a][b][self.count] += 1
            else:
                train_dict[a][b][c] += 1
                train_dict[a][b][self.count] += 1
        self.word_dict = train_dict
        self.train_data = data

    def predict(self, data):

        number_to_word = data.get_word_dict_tran()
        word_to_number = self.train_data.get_word_dict()
        out_list = []
        for key in data:
            a, b, c = key
            if number_to_word[a] not in word_to_number:
                # out_list.append(0)
                continue
            a = word_to_number[number_to_word[a]]
            if number_to_word[b] not in word_to_number:
                # out_list.append(0)
                continue
            b = word_to_number[number_to_word[b]]
            if number_to_word[c] not in word_to_number:
                # out_list.append(0)
                continue
            c = word_to_number[number_to_word[c]]
            if a not in self.word_dict:
                # out_list.append(0)
                continue
            elif b not in self.word_dict[a]:
                # out_list.append(0)
                continue
            elif c not in self.word_dict[a][b]:
                # out_list.append(0)
                continue
            else:
                # print(self.word_dict[a][b][c], self.word_dict[a][b][self.count])
                out_list.append(self.word_dict[a][b][c]/self.word_dict[a][b][self.count])
        return out_list

    def predict_single(self, word1, word2):

        word_to_number = self.train_data.get_word_dict()
        number_to_word = self.train_data.get_word_dict_tran()
        word = ''
        max_count = 0
        # print(self.word_dict[word_to_number[word1]][word_to_number[word2]])
        for key in self.word_dict[word_to_number[word1]][word_to_number[word2]]:
            if key != self.count and self.word_dict[word_to_number[word1]][word_to_number[word2]][key] > max_count:
                # print(key, self.word_dict[word_to_number[word1]][word_to_number[word2]][key])
                max_count = self.word_dict[word_to_number[word1]][word_to_number[word2]][key]
                word = key
        return number_to_word[word]

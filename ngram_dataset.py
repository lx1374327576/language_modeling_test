import os


class NgramDataset(object):

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __init__(self, file_name=None):
        if not file_name:
            file_name = os.path.join(os.getcwd(), 'data/penn/train.txt')
        f = open(file_name, 'r')
        f_str = f.read()
        f_str = f_str.replace('.', ' DOT')
        f_str = f_str.replace(',', ' DOT')
        f_str = f_str.replace('\n', '')
        word_list = f_str.split(' ')
        new_list = []
        for word in word_list:
            if len(word) != 0:
                new_list.append(word)
        self.word_list = new_list


class TrigramDataset(NgramDataset):

    def __init__(self, file_name=None, mode='train'):
        if not file_name:
            file_name = os.path.join(os.getcwd(), 'data/penn/train.txt')
        NgramDataset.__init__(file_name)
        word_dict = {}
        word_dict_tran = {}
        word_count = 0
        for word in self.word_list:
            if word not in word_dict:
                word_count += 1
                word_dict[word] = word_count
                word_dict_tran[word_count] = word
        self.word_dict = word_dict
        self.word_dict_tran = word_dict_tran
        new_word_list = []
        for i in range(len(self.word_list)):
            new_word_list.append(word_dict[self.word_list[i]])
        data = []
        for i in range(len(new_word_list)):
            data.append((new_word_list[i], new_word_list[i+1], new_word_list[i+2]))
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get_word_dict(self):
        return self.word_dict

    def get_word_dict_tran(self):
        return self.word_dict_tran

#import tensorflow as tf
import create_dict
import random
import codecs, csv

class MyData:
    def __init__(self, csv_name, max_freq=1.0, min_freq=0.0):
        self.csv_file_name = csv_name
        self.word_dict, self.label_dict = create_dict.createDict(csv_name, max_freq, min_freq)
        self.word_dict_size = len(self.word_dict)
        self.label_dict_size = len(self.label_dict)
        self.label_list = [ k for k, v in sorted(self.label_dict.items(), key=lambda x: x[1])]
        self.current_idx = 0

    def import_data(self, csv_name, seed=None):
        print("reading csv")
        csvfile = codecs.open(csv_name or self.csv_file_name, 'r', encoding="utf-8")
        csvReader = csv.reader(csvfile)
        self.csv_data = [ row for row in csvReader ]
        self.csv_data_rand = list( enumerate(self.csv_data) ) # [ id, row ]
        csvfile.close()
        random.seed(seed, version=2)
        random.shuffle(self.csv_data_rand)
        self.data_size = len(self.csv_data_rand)

    def read_rewind(self):
       self.current_idx = 0

    def read_next(self):
       if self.current_idx >= len(self.csv_data_rand):
           return None
       i, row = self.csv_data_rand[self.current_idx]
       self.current_idx += 1
       word_vec, label_dec = create_dict.vectorize(row, self.word_dict, self.label_dict)
       return i, word_vec, label_dec

    def read_next_batch(self, batch_size):
       id_batch = []
       word_batch = []
       label_batch = []
       for x in range(batch_size):
           i, w, l = self.read_next()
           id_batch.append(i)
           word_batch.append(w)
           label_batch.append(l)
       return id_batch, word_batch, label_batch


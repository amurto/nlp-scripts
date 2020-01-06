import pandas as pd
import os
import time
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel
import numpy as np
import nltk
from utils import pickle_dump, pickle_load
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from docopt import docopt
import sys
import codecs

def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, encoding="utf8") as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            word_vector = np.array([float(v) for v in line[1:]])
            word_vectors[word] = word_vector
            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)

    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())

    return word_vectors, embeddings_dim

def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result

def build_vocabulary(corpus, start_id=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_id))

def build_embedding(corpus, vocab, embedding_dim=300):
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, embedding_dim), dtype='float32')

    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    #print('embedding out of vocabulary:', str(count))
    return emb

def build_glove_embedding(vocab, word_vectors, embed_dim):
    emb_matrix = np.zeros(shape=(len(vocab) + 2, embed_dim), dtype='float32')

    count = 0
    for word, i in vocab.items():
        if word not in word_vectors:
            count += 1
            emb_matrix[i, :] = np.random.uniform(-0.1, 0.1, embed_dim)
        else:
            emb_matrix[i, :] = word_vectors[word]
    #print('glove embedding out of vocabulary:', str(count))
    return emb_matrix

def build_aspect_embedding(aspect_vocab, split_func, word_vocab, word_embed):
    aspect_embed = np.random.uniform(-0.1, 0.1, [len(aspect_vocab.keys()), word_embed.shape[1]])
    count = 0
    for aspect, aspect_id in aspect_vocab.items():
        word_ids = [word_vocab.get(word, 0) for word in split_func(aspect)]
        if any(word_ids):
            avg_vector = np.mean(word_embed[word_ids], axis=0)
            aspect_embed[aspect_id] = avg_vector
        else:
            count += 1
    #print('aspect embedding out of vocabulary:', str(count))
    return aspect_embed

def build_aspect_text_embedding(aspect_text_vocab, word_vocab, word_embed):
    aspect_text_embed = np.zeros(shape=(len(aspect_text_vocab) + 2, word_embed.shape[1]), dtype='float32')
    count = 0
    for aspect, aspect_id in aspect_text_vocab.items():
        if aspect in word_vocab:
            aspect_text_embed[aspect_id] = word_embed[word_vocab[aspect]]
        else:
            count += 1
            aspect_text_embed[aspect_id] = np.random.uniform(-0.1, 0.1, word_embed.shape[1])
    #print('aspect text embedding out of vocabulary:', str(count))
    return aspect_text_embed

def get_loc_info(l, start, end):
    pos_info = []
    offset_info =[]
    for i in range(len(l)):
        if i < start:
            pos_info.append(1 - abs(i - start) / len(l))
            offset_info.append(i - start)
        elif start <= i < end:
            pos_info.append(1.)
            offset_info.append(0.)
        else:
            pos_info.append(1 - abs(i - end + 1) / len(l))
            offset_info.append(i - end +1)
    return pos_info, offset_info

def split_text_and_get_loc_info( word_vocab, char_vocab, word_cut_func,start,end):
    word_input_l, word_input_r, word_input_r_with_pad, word_pos_input, word_offset_input = [], [], [], [], []
    char_input_l, char_input_r, char_input_r_with_pad, char_pos_input, char_offset_input = [], [], [], [], []
    word_mask, char_mask = [], []
    #for idx, row in data.iterrows():
    test_data_word_list = word_cut_func(test_data_content)
    test_data_char_list = test_data_content
    text, word_list, char_list, aspect = test_data_content, test_data_word_list, test_data_char_list, test_data_aspect
    #start, end = row['from'], row['to']
    start=start
    end=end
    char_input_l.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[:end])))
    char_input_r.append(list(map(lambda x: char_vocab.get(x, len(char_vocab)+1), char_list[start:])))
    char_input_r_with_pad.append([char_vocab.get(char_list[i], len(char_vocab)+1) if i >=0 else 0
                                          for i in range(len(char_list))])  # replace left sequence with 0
    _char_mask = [1] * len(char_list)
    _char_mask[start:end] = [0.5] * (end-start)     # 1 for context, 0.5 for aspect
    char_mask.append(_char_mask)
    _pos_input, _offset_input = get_loc_info(char_list, start, end)
    char_pos_input.append(_pos_input)
    char_offset_input.append(_offset_input)

    word_list_l = word_cut_func(text[:start])
    word_list_r = word_cut_func(text[end:])
    start = len(word_list_l)
    end = len(word_list) - len(word_list_r)
    if word_list[start:end] != word_cut_func(aspect):
        if word_list[start-1:end] == word_cut_func(aspect):
            start -= 1
        elif word_list[start:end+1] == word_cut_func(aspect):
            end += 1
        else:
            raise Exception('Can not find aspect `{}` in `{}`, word list : `{}`'.format(aspect, text, word_list))
    word_input_l.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[:end])))
    word_input_r.append(list(map(lambda x: word_vocab.get(x, len(word_vocab)+1), word_list[start:])))
    word_input_r_with_pad.append([word_vocab.get(word_list[i], len(word_vocab) + 1) if i >= start else 0
                                  for i in range(len(word_list))])      # replace left sequence with 0
    _word_mask = [1] * len(word_list)
    _word_mask[start:end] = [0.5] * (end - start)  # 1 for context, 0.5 for aspect
    word_mask.append(_word_mask)
    _pos_input, _offset_input = get_loc_info(word_list, start, end)
    word_pos_input.append(_pos_input)
    word_offset_input.append(_offset_input)
    return (word_input_l, word_input_r, word_input_r_with_pad, word_mask, word_pos_input, word_offset_input,
            char_input_l, char_input_r, char_input_r_with_pad, char_mask, char_pos_input, char_offset_input)


#test_input2="John thinks the food is very bad"
def pre_process(file_folder, word_cut_func, is_en,start,end):
    # test_data = pd.read_csv(os.path.join('./data/twitter', 'test.csv'), header=0, index_col=None,encoding = 'unicode_escape')
    # print(type(test_data['content']))
    #test_data = pd.read_csv('data\sample\sample_data.csv', header=0, index_col=None,encoding = 'unicode_escape')
    #print(test_data)
    print('preprocessing: ', file_folder)
    
    test_data_word_list = word_cut_func(test_data_content)
    test_data_char_list = test_data_content
    test_data_aspect_word_list = word_cut_func(test_data_aspect)
    test_data_aspect_char_list = test_data_aspect
    #train_data['aspect_word_list'] = train_data['aspect'].apply(word_cut_func)
    #train_data['aspect_char_list'] = train_data['aspect'].apply(lambda x: list(x))
    print('building vocabulary...')
    word_corpus = (test_data_word_list)
    char_corpus = (test_data_char_list)
    aspect_corpus = (test_data_aspect)
    aspect_text_word_corpus = test_data_aspect_word_list
    aspect_text_char_corpus = test_data_aspect_char_list

    word_vocab = build_vocabulary(word_corpus, start_id=1)
    char_vocab = build_vocabulary(char_corpus, start_id=1)
    aspect_vocab = build_vocabulary(aspect_corpus, start_id=0)
    aspect_text_word_vocab = build_vocabulary(aspect_text_word_corpus, start_id=1)
    aspect_text_char_vocab = build_vocabulary(aspect_text_char_corpus, start_id=1)
    #print(word_vocab.get(word, len(word_vocab)+1) for word in (word_vocab))

    pickle_dump(word_vocab, os.path.join(file_folder, 'word_vocab2.pkl'))
    pickle_dump(char_vocab, os.path.join(file_folder, 'char_vocab2.pkl'))
    pickle_dump(aspect_vocab, os.path.join(file_folder, 'aspect_vocab2.pkl'))
    pickle_dump(aspect_text_word_vocab, os.path.join(file_folder, 'aspect_text_word_vocab2.pkl'))
    pickle_dump(aspect_text_char_vocab, os.path.join(file_folder, 'aspect_text_char_vocab2.pkl'))

    # prepare embedding
    print('preparing embedding...')
    word_w2v = build_embedding(word_corpus, word_vocab, config.word_embed_dim)
    aspect_word_w2v = build_aspect_embedding(aspect_vocab, word_cut_func, word_vocab, word_w2v)
    aspect_text_word_w2v = build_aspect_text_embedding(aspect_text_word_vocab, word_vocab, word_w2v)
    char_w2v = build_embedding(char_corpus, char_vocab, config.word_embed_dim)
    aspect_char_w2v = build_aspect_embedding(aspect_vocab, lambda x: list(x), char_vocab, char_w2v)
    aspect_text_char_w2v = build_aspect_text_embedding(aspect_text_char_vocab, char_vocab, char_w2v)
    np.save(os.path.join(file_folder, 'word_w2v.npy'), word_w2v)
    np.save(os.path.join(file_folder, 'aspect_word_w2v.npy'), aspect_word_w2v)
    np.save(os.path.join(file_folder, 'aspect_text_word_w2v.npy'), aspect_text_word_w2v)
    np.save(os.path.join(file_folder, 'char_w2v.npy'), char_w2v)
    np.save(os.path.join(file_folder, 'aspect_char_w2v.npy'), aspect_char_w2v)
    np.save(os.path.join(file_folder, 'aspect_text_char_w2v.npy'), aspect_text_char_w2v)
    print('finished preparing embedding!')

    if is_en:
        word_glove = build_glove_embedding(word_vocab, glove_vectors, glove_embed_dim)
        aspect_word_glove = build_aspect_embedding(aspect_vocab, word_cut_func, word_vocab, word_glove)
        aspect_text_word_glove = build_aspect_text_embedding(aspect_text_word_vocab, word_vocab, word_glove)
        np.save(os.path.join(file_folder, 'word_glove.npy'), word_glove)
        np.save(os.path.join(file_folder, 'aspect_word_glove.npy'), aspect_word_glove)
        np.save(os.path.join(file_folder, 'aspect_text_word_glove.npy'), aspect_text_word_glove)
        #print('shape of word_glove:', word_glove.shape)
        #print('sample of word_glove:', word_glove[:2, :5])
        #print('shape of aspect_word_glove:', aspect_word_glove.shape)
        #print('sample of aspect_word_glove:', aspect_word_glove[:2, :5])
        #print('shape of aspect_text_word_glove:', aspect_text_word_glove.shape)
        #print('sample of aspect_text_word_glove:', aspect_text_word_glove[:2, :5])

    # prepare input
    print('preparing text input...')
    g=lambda x: [word_vocab.get(word, len(word_vocab)+1) for word in x]
    f=lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]
    test_word_input = g(test_data_word_list)
    test_char_input = f(test_data_char_list)
     
    pickle_dump(test_word_input, os.path.join(file_folder, 'test_word_input2.pkl'))
    pickle_dump(test_char_input, os.path.join(file_folder, 'test_char_input2.pkl'))
    print('finished preparing text input!')

    print('preparing aspect input...')
    #train_aspect_input = train_data['aspect'].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    #valid_aspect_input = valid_data['aspect'].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    test_aspect_input = test_data_aspect
    #pickle_dump(train_aspect_input, os.path.join(file_folder, 'train_aspect_input.pkl'))
    #pickle_dump(valid_aspect_input, os.path.join(file_folder, 'valid_aspect_input.pkl'))
    pickle_dump(test_aspect_input, os.path.join(file_folder, 'test_aspect_input.pkl'))
    print('finished preparing aspect input!')

    print('preparing aspect text input...')
    x=lambda x: [aspect_text_word_vocab.get(word, len(aspect_text_word_vocab) + 1) for word in x]
    y=lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]
    test_aspect_text_word_input =x(test_data_aspect_word_list)
    test_aspect_text_char_input =y(test_data_aspect_char_list)

    pickle_dump(test_aspect_text_word_input, os.path.join(file_folder, 'test_word_aspect_input.pkl'))
    pickle_dump(test_aspect_text_char_input, os.path.join(file_folder, 'test_char_aspect_input.pkl'))
    print('finished preparing aspect text input!')

    test_word_input_l, test_word_input_r, test_word_input_r_with_pad, test_word_mask, test_word_pos_input, \
            test_word_offset_input, test_char_input_l, test_char_input_r, test_char_input_r_with_pad, test_char_mask, \
            test_char_pos_input, test_char_offset_input = split_text_and_get_loc_info(word_vocab,
                                                                                      char_vocab, word_cut_func,start,end)
    pickle_dump(test_word_input_l, os.path.join(file_folder, 'test_word_input_l2.pkl'))
    pickle_dump(test_word_input_r, os.path.join(file_folder, 'test_word_input_r2.pkl'))
    pickle_dump(test_word_input_r_with_pad, os.path.join(file_folder, 'test_word_input_r_with_pad2.pkl'))
    pickle_dump(test_word_mask, os.path.join(file_folder, 'test_word_mask2.pkl'))
    pickle_dump(test_word_pos_input, os.path.join(file_folder, 'test_word_pos_input2.pkl'))
    pickle_dump(test_word_offset_input, os.path.join(file_folder, 'test_word_offset_input2.pkl'))
    pickle_dump(test_char_input_l, os.path.join(file_folder, 'test_char_input_l2.pkl'))
    pickle_dump(test_char_input_r, os.path.join(file_folder, 'test_char_input_r2.pkl'))
    pickle_dump(test_char_input_r_with_pad, os.path.join(file_folder, 'test_char_input_r_with_pad2.pkl'))
    pickle_dump(test_char_mask, os.path.join(file_folder, 'test_char_mask2.pkl'))
    pickle_dump(test_char_pos_input, os.path.join(file_folder, 'test_char_pos_input2.pkl'))
    pickle_dump(test_char_offset_input, os.path.join(file_folder, 'test_char_offset_input2.pkl'))

    # prepare output
    #if 'sentiment' in test_data.columns:
    #    pickle_dump(test_data['sentiment'].values.tolist(), os.path.join(file_folder, 'test_label.pkl'))
    print('finished preparing output!')


def load_input_data2(data_folder, data_kind, level, use_text_input, use_text_input_l, use_text_input_r,
                    use_text_input_r_with_pad, use_aspect_input, use_aspect_text_input, use_loc_input,
                    use_offset_input, use_mask):
    dirname = os.path.join('./data', data_folder)
    input_data = []
    
    #input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input2.pkl'.format(data_kind, level))))
    if use_text_input:
       input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input2.pkl'.format(data_kind, level))))
    if use_text_input_l:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_l2.pkl'.format(data_kind, level))))
    if use_text_input_r:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_r2.pkl'.format(data_kind, level))))
    if use_text_input_r_with_pad:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input_r_with_pad2.pkl'.format(data_kind, level))))
    if use_aspect_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_aspect_input2.pkl'.format(data_kind))))
    if use_aspect_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_aspect_input2.pkl'.format(data_kind, level))))
    if use_loc_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_pos_input2.pkl'.format(data_kind, level))))
    if use_offset_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_offset_input2.pkl'.format(data_kind, level))))
    if use_mask:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_mask2.pkl'.format(data_kind, level))))
    if len(input_data) == 1:
        input_data = input_data[0]
    if len(input_data) == 0:
        raise Exception('No Input!')
    print(input_data)
    return input_data

def train_model(data_folder, data_name, level, model_name, is_aspect_term=True):
    config.data_folder = data_folder
    config.data_name = data_name
    if not os.path.exists(os.path.join(config.checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(config.checkpoint_dir, data_folder))
    config.level = level
    config.model_name = model_name
    config.is_aspect_term = is_aspect_term
    config.init_input()
    config.exp_name = '{}_{}_wv_{}'.format(model_name, level, config.word_embed_type)
    config.exp_name = config.exp_name + '_update' if config.word_embed_trainable else config.exp_name + '_fix'
    if config.use_aspect_input:
        config.exp_name += '_aspv_{}'.format(config.aspect_embed_type)
        config.exp_name = config.exp_name + '_update' if config.aspect_embed_trainable else config.exp_name + '_fix'
    if config.use_elmo:
        config.exp_name += '_elmo_alone_{}_mode_{}_{}'.format(config.use_elmo_alone, config.elmo_output_mode,
                                                              'update' if config.elmo_trainable else 'fix')

    #print(config.exp_name)
    model = SentimentModel(config)

    #test_input = load_input_data2(data_folder, 'test', level, config.use_text_input)
    test_input = load_input_data2(data_folder, 'test', level, config.use_text_input, config.use_text_input_l,
                                 config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
                                 config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
                                 config.use_mask)   
    
    #test_label = load_label(data_folder, 'test')
    #load the best model
    model.load()
    
    # print('score over valid data...')
    # model.score(valid_input, valid_label)
    #print('score over test data...')
    model.score2(test_input)


if __name__ == '__main__':
    config = Config()
    config.use_elmo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False

    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
    glove_vectors, glove_embed_dim = load_glove_format('./raw_data/glove.42B.300d.txt')
    test_data_content= input("Enter Sentence:")
    test_data_aspect=input("Enter aspect that is to be analysed:")
    #test_data_content="The food was really bad"
    #test_data_aspect="food"
    start=test_data_content.lower().find(test_data_aspect.lower())
    end=start+len(test_data_aspect)
    #print("Start:", start)
    #print("End:",end)
    pre_process('./data/sample', lambda x: nltk.word_tokenize(x), True,start,end)
    train_model('twitter', 'twitter', 'word', 'td_lstm')
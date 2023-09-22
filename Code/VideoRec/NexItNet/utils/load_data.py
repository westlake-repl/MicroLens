import os
import torch
import numpy as np

def read_items(args):
    item_id_to_keys = {}
    item_name_to_id = {}
    for i in range(args.min_video_no, args.max_video_no + 1):
        image_name = str(i)
        item_id = i
        item_name_to_id[image_name] = item_id
        item_id_to_keys[item_id] = image_name
    return item_id_to_keys, item_name_to_id

def read_texts(tokenizer, args):
    text_path = os.path.join(args.root_data_dir, args.dataset, args.text_data)
    item_dic = {}
    item_name_to_index = {}
    item_index_to_name = {}
    index = 1

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            splited = line.strip('\n').split(',')
            doc_name, title = splited[0], str(','.join(splited[1:]))
            # if 'scale' in args.dataset:
            #     splited = line.strip('\n').split(',')
            #     doc_name, title = splited[0], str(','.join(splited[1:]))
            # elif 'MIND' in args.dataset:
            #     splited = line.strip('\n').split('\t')
            #     doc_name, title, _ = splited
            # else:
            #     splited = line.strip('\n').split('\t')
            #     doc_name, title = splited
            item_name_to_index[doc_name] = index
            item_index_to_name[index] = doc_name
            index += 1
            # tokenizer
            tokenized_title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
            item_dic[doc_name] = [tokenized_title]

    return item_dic, item_name_to_index, item_index_to_name

def read_videos(min_video_no, max_video_no):
    item_id_to_keys = {}
    item_name_to_id = {}
    for i in range(min_video_no, max_video_no + 1):
        image_name = str(i)
        item_id = i
        item_name_to_id[image_name] = item_id
        item_id_to_keys[item_id] = image_name
    return item_id_to_keys, item_name_to_id

def read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args):
    behaviors_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    max_seq_len, min_seq_len = args.max_seq_len, args.min_seq_len

    Log_file.info('##### item number {}'.format(len(before_item_id_to_keys)))
    Log_file.info('##### min seq len {}, max seq len {}#####'.format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, 'r') as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            history_item_name = str(splited[1]).strip().split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[str(i)] for i in history_item_name]
            user_seq_dic[user_id] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1

    Log_file.info("##### pairs_num {}".format(pairs_num))
    Log_file.info('##### user seqs before {}'.format(before_seq_num))

    item_id = 1
    item_id_to_keys = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_keys[item_id] = before_item_id_to_keys[before_item_id]
            item_id += 1

    item_num = len(item_id_before_to_now)
    Log_file.info('##### items after clearing {}, {}, {}, {}#####'.format(item_num, item_id - 1, len(item_id_to_keys), len(item_id_before_to_now)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    
    if args.power < 0:
        train_item_counts = [1] * (item_num + 1)
    else:
        train_item_counts = [0] * (item_num + 1)
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]

        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]

        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        for i in train:
            train_item_counts[i] += 1

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
        
    item_counts_powered = np.power(train_item_counts, args.power)
    pop_prob_list = []
    for i in range(1, item_num + 1):
        pop_prob_list.append(item_counts_powered[i])
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)
    
    Log_file.info('prob max: {}, prob min: {}, prob mean: {}'.\
        format(max(pop_prob_list), min(pop_prob_list), np.mean(pop_prob_list)))
    Log_file.info('##### user seqs after clearing {}, {}, {}, {}#####'.format(seq_num, len(user_seq_dic), len(users_train), len(users_valid)))
    if args.mode =='train':
        return item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, pop_prob_list
    return item_num, item_id_to_keys, users_train, users_test, users_history_for_test, pop_prob_list


def read_behaviors_text(item_dic, before_item_name_to_index, before_item_index_to_name, Log_file, args):
    behaviors_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    max_seq_len, min_seq_len = args.max_seq_len, args.min_seq_len

    Log_file.info('##### text number {} {} {} (before clearing)#####'.format(len(before_item_name_to_index), len(item_dic), len(before_item_index_to_name)))
    Log_file.info('##### min seq len {}, max seq len {}#####'.format(min_seq_len, max_seq_len))

    before_item_num = len(before_item_name_to_index)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    before_seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, 'r') as f:
        for line in f:
            before_seq_num += 1
            splited = line.strip('\n').split('\t')
            user_id = splited[0]
            history_item_name = splited[1].split(' ')

            if len(history_item_name) < min_seq_len:
                continue

            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_index[i] for i in history_item_name]

            user_seq_dic[user_id] = history_item_name
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    
    Log_file.info("##### pairs_num {}".format(pairs_num))
    Log_file.info('#### user seqs before {}'.format(before_seq_num))

    for item_id in range(1, before_item_num + 1):
        if before_item_counts[item_id] == 0:
            item_dic.pop(before_item_index_to_name[item_id])

    item_id = 1
    item_num = len(item_dic)
    item_index = {}

    for doc_name, value in item_dic.items():
        item_index[doc_name] = item_id
        item_id += 1

    Log_file.info('##### items after clearing {}, {}#####'.format(item_num, len(item_index)))
    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    
    if args.power < 0:
        train_item_counts = [1] * (item_num + 1)
    else:
        train_item_counts = [0] * (item_num + 1)
    for _, user_seq_name in user_seq_dic.items():
        user_seq = [item_index[item_name] for item_name in user_seq_name]
        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]

        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        for i in train:
            train_item_counts[i] += 1

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))

        user_id += 1
    
    item_counts_powered = np.power(train_item_counts, args.power)
    pop_prob_list = []
    for i in range(1, item_num + 1):
        pop_prob_list.append(item_counts_powered[i])
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)
    
    Log_file.info('prob max: {}, prob min: {}, prob mean: {}'.\
        format(max(pop_prob_list), min(pop_prob_list), np.mean(pop_prob_list)))
    Log_file.info('##### user seqs after clearing {}, {}, {}, {}#####'.format(seq_num, len(user_seq_dic), len(users_train), len(users_valid)))
    if args.mode =='train':
        return item_num, item_dic, item_index, users_train, users_valid, users_history_for_valid, pop_prob_list
    return item_num, item_dic, item_index, users_train, users_test, users_history_for_test, pop_prob_list

def get_doc_input_bert(text_dic, item_index, args):
    item_num = len(text_dic) + 1

    news_title = np.zeros((item_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')

    for key in text_dic:
        title = text_dic[key]
        doc_index = item_index[key]
        
        news_title[doc_index] = title[0]['input_ids']
        news_title_attmask[doc_index] = title[0]['attention_mask']

    return news_title, news_title_attmask

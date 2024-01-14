# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


project = 'HVIDEO'
path_to_raw = '../data/'+project+'/raw/'
path_to_processed = '../data/'+project+'/'

files = ['traindata.txt','validdata.txt','testdata.txt']
domains = {'E':[],
           'V':[]
          }

print('loading data...')
all_data = pd.DataFrame(columns=['user_id','history'])
for f in files:

    string_input = []
    with open(path_to_raw+f, 'r') as myfile:
        print('loading file: %s...'%(path_to_raw+f))

        for line in tqdm(myfile.readlines()):
            user_id = line.split('\t')[0]
            line = line.replace('||\t', '\n'+user_id+'\t').replace('||','')
            string_input.append(line)

    string_input = ''.join(string_input)
    df = pd.read_csv(StringIO(string_input), sep='\t', header=None, names=['user_id','history'])

    all_data = pd.concat([all_data, df], axis=0)

all_data['domain'] = all_data['history'].apply(lambda x: x.split('|')[0][0])
all_data['item_id'] = all_data['history'].apply(lambda x: x.split('|')[0][1:])
all_data['timestamp'] = all_data['history'].apply(lambda x: x.split('|')[3])

all_data.drop('history', axis=1, inplace=True)
all_data.drop_duplicates(inplace=True)


def create_all_data(
    df,
    output_path,
    output_name='all_data.txt',
    convert_dtypes=False,
    save_result=False,
    return_result=True,
):
    # part 1
    data = []
    for _, row in tqdm(df.iterrows()):
        data.append({
            'user_id': int(row.user_id) if convert_dtypes else row.user_id,
            'item_id':  int(row.item_id) if convert_dtypes else row.item_id,
            'timestamp': int(row.timestamp) if convert_dtypes else row.timestamp
        })


    # part 2
    user_history = defaultdict(list)
    item_history = defaultdict(list)

    for row in tqdm(data):
        user_raw_id = row['user_id']
        item_raw_id = row['item_id']
        interaction_timestamp = row['timestamp']

        user_history[user_raw_id].append({'item_id': item_raw_id, 'timestamp': interaction_timestamp})
        item_history[item_raw_id].append({'user_id': user_raw_id, 'timestamp': interaction_timestamp})


    # part 3
    is_changed = True
    threshold = 5
    good_users = set()
    good_items = set()

    while is_changed:
        old_state = (len(good_users), len(good_items))

        good_users = set()
        good_items = set()

        for user_id, history in user_history.items():
            if len(history) >= threshold:
                good_users.add(user_id)

        for item_id, history in item_history.items():
            if len(history) >= threshold:
                good_items.add(item_id)

        user_history = {
            user_id: list(filter(lambda x: x['item_id'] in good_items, history))
            for user_id, history in user_history.items()
        }

        item_history = {
            item_id: list(filter(lambda x: x['user_id'] in good_users, history))
            for item_id, history in item_history.items()
        }

        new_state = (len(good_users), len(good_items))
        is_changed = (old_state != new_state)
        print(old_state, new_state)


    # part 4
    user_mapping = {}
    item_mapping = {}
    tmp_user_history = defaultdict(list)
    tmp_item_history = defaultdict(list)

    for user_id, history in tqdm(user_history.items()):
        processed_history = []

        for filtered_item in history:
            item_id = filtered_item['item_id']
            item_timestamp = filtered_item['timestamp']

            processed_item_id = item_mapping.get(item_id, len(item_mapping) + 1)
            item_mapping[item_id] = processed_item_id

            processed_history.append({'item_id': processed_item_id, 'timestamp': item_timestamp})

        if len(processed_history) >= threshold:
            processed_user_id = user_mapping.get(user_id, len(user_mapping) + 1)
            user_mapping[user_id] = processed_user_id

            tmp_user_history[processed_user_id] = sorted(processed_history, key=lambda x: x['timestamp'])


    for item_id, history in tqdm(item_history.items()):
        processed_history = []

        for filtered_user in history:
            user_id = filtered_user['user_id']
            user_timestamp = filtered_user['timestamp']

            processed_user_id = user_mapping.get(user_id, len(user_mapping) + 1)
            user_mapping[user_id] = processed_user_id

            processed_history.append({'user_id': processed_user_id, 'timestamp': user_timestamp})

        if len(processed_history) >= threshold:
            processed_item_id = item_mapping.get(item_id, len(item_mapping) + 1)
            item_mapping[item_id] = processed_item_id

            tmp_item_history[processed_item_id] = sorted(processed_history, key=lambda x: x['timestamp'])

    user_history = tmp_user_history
    item_history = tmp_item_history

    # part 5
    if save_result:
        with open(output_path+'/'+output_name, 'w') as f:
            for user_id, item_history in user_history.items():
                f.write(' '.join([str(user_id)] + [
                    str(item_event['item_id']) for item_event in sorted(item_history, key=lambda x: x['timestamp'])
                ]))
                f.write('\n')

    if return_result:
        return user_history, item_history
    return None, None

print('preprocessing and saving data...')
for d in tqdm(domains):
    new_dir = path_to_processed + d
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    df = all_data[all_data['domain']==d].drop('domain', axis=1).reset_index(drop=True)

    user_history, item_history = create_all_data(df, new_dir)
    for output_name in ['all_data.txt']:
        with open(new_dir+'/'+output_name, 'w') as f:
                for user_id, item_history in user_history.items():
                    f.write(' '.join([str(user_id)] + [
                        str(item_event['item_id']) for item_event in sorted(item_history, key=lambda x: x['timestamp'])
                    ]))
                    f.write('\n')

print('script finished.')

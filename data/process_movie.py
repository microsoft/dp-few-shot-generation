# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Assume saving data from https://github.com/tonyzhaozh/few-shot-learning/tree/main/data/slot-movies in the current folder

import pandas as pd
import json
import pickle
import numpy as np
import os


# Imported from https://github.com/tonyzhaozh/few-shot-learning/blob/main/data_utils.py#L122-#L173
def generate_dataset(field_name, data_path="."):

    all_fields = ["Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
    assert field_name in all_fields
    all_fields.remove(field_name)
    filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
    target_tags = [f"B-{field_name}", f"I-{field_name}"]

    with open(f'{data_path}/slot-movies/train', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    train_answers = []
    train_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            train_answers.append(answer.strip())
            train_sentences.append(untagged_line.strip())


    with open(f'{data_path}/slot-movies/test', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    test_answers = []
    test_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            test_answers.append(answer.strip())
            test_sentences.append(untagged_line.strip())
    if not os.path.isdir((f"{data_path}/movie/{field_name}")):
        os.makedirs(f"{data_path}/movie/{field_name}", exist_ok=True)
    train_data = {}
    train_data['content'] = train_sentences
    train_data['label'] = train_answers
    df = pd.DataFrame(train_data)
    df.to_csv(f"{data_path}/movie/{field_name}/train.csv")

    test_data = {}
    test_data['content'] = test_sentences
    test_data['label'] = test_answers
    df = pd.DataFrame(test_data)
    df.to_csv(f"{data_path}/movie/{field_name}/test.csv")

for field_name in ['Director', 'Genre']:
    # by default save to ./movie/field_name, this is consistent with src/run_exp_movie.py#L362 
    generate_dataset(field_name, data_path='./')
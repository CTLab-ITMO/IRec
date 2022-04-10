from .base import BaseDataset

from transformers import BertTokenizer  # TODO remove from here

import os
import wget
import zipfile
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class ColaDataset(BaseDataset, config_name='cola'):
    URL = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    def __init__(self, dataset_dir_path):
        archive_upload_path = '{}/cola_public_1.1.zip'.format(dataset_dir_path)
        unzipped_dir_path = '{}/cola_public'.format(dataset_dir_path)

        if not os.path.exists(archive_upload_path):
            wget.download(ColaDataset.URL, archive_upload_path)

            if not os.path.exists(unzipped_dir_path):
                with zipfile.ZipFile(archive_upload_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir_path)

        df = pd.read_csv(
            '{}/raw/in_domain_train.tsv'.format(unzipped_dir_path),
            delimiter='\t',
            header=None,
            names=['sentence_source', 'label', 'label_notes', 'sentence']
        )

        # TODO convert to processor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sentences = df.sentence.values
        self.labels = df.label.values

        logger.info('ColaDataset has been created!')

    def __getitem__(self, item):
        sentence = self.sentences[item]
        encoded_dict = self.tokenizer.encode_plus(  # TODO convert into processor
            sentence,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        tokens = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
            'label': self.labels[item]
        }

    def __len__(self):
        return len(self.labels)

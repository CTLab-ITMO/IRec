from .base import BaseDataset

import os
import zipfile
import pandas as pd
import urllib.request

import logging

logger = logging.getLogger(__name__)


class ColaDataset(BaseDataset, config_name='cola'):
    URL = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    def __init__(self, dataset_dir_path):

        archive_upload_path = '{}/cola_public_1.1.zip'.format(dataset_dir_path)
        unzipped_dir_path = '{}/cola_public'.format(dataset_dir_path)

        if not os.path.exists(dataset_dir_path):
            os.mkdir(dataset_dir_path)

        if not os.path.exists(archive_upload_path):
            with urllib.request.urlopen(ColaDataset.URL) as response, open(archive_upload_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

            if not os.path.exists(unzipped_dir_path):
                with zipfile.ZipFile(archive_upload_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir_path)

        df = pd.read_csv(
            '{}/raw/in_domain_train.tsv'.format(unzipped_dir_path),
            delimiter='\t',
            header=None,
            names=['sentence_source', 'label', 'label_notes', 'sentence']
        )

        self.sentences = df.sentence.values
        self.labels = df.label.values

        logger.info('ColaDataset has been created!')

    def __getitem__(self, item):
        return {
            'sample': self.sentences[item],
            'label': self.labels[item]
        }

    def __len__(self):
        return len(self.labels)

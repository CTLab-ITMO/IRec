from .base import TorchEncoder

from transformers import BertForSequenceClassification

BERT_CLS = {
    'classification': BertForSequenceClassification
}


class BertEncoder(TorchEncoder, config_name='bert'):

    def __init__(
            self,
            task_type,
            input_ids_prefix,
            attention_mask_prefix,
            ground_truth_prefix,
            loss_prefix,
            predictions_prefix,
            bert_kwargs=None
    ):
        super().__init__()
        self._input_ids_prefix = input_ids_prefix
        self._attention_mask_prefix = attention_mask_prefix
        self.ground_truth_prefix = ground_truth_prefix
        self._loss_prefix = loss_prefix
        self._predictions_prefix = predictions_prefix

        if bert_kwargs is None:
            bert_kwargs = dict()

        self._encoder = BERT_CLS[task_type].from_pretrained(**bert_kwargs)

    def __call__(self, inputs):
        loss, logits = self._encoder(
            input_ids=inputs[self._input_ids_prefix],
            attention_mask=inputs[self._attention_mask_prefix],
            token_type_ids=None,
            labels=inputs[self.ground_truth_prefix],
            return_dict=False
        )
        inputs[self._loss_prefix] = loss
        inputs[self._predictions_prefix] = logits
        return inputs

    @classmethod
    def create_from_config(cls, config, schema=None):
        assert schema is not None, 'Schema should be provided'
        return cls(
            task_type=config['task_type'],
            input_ids_prefix=schema['inputs_prefix'],
            ground_truth_prefix=schema['ground_truth_prefix'],
            attention_mask_prefix=schema['attention_mask_prefix'],
            loss_prefix=schema['loss_prefix'],
            predictions_prefix=schema['predictions_prefix'],
            bert_kwargs=config.get('bert_kwargs', None)
        )

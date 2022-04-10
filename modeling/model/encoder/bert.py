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
            labels_prefix,
            loss_prefix='loss',
            logits_prefix='logits',
            bert_kwargs=None
    ):
        super().__init__()
        self._input_ids_prefix = input_ids_prefix
        self._attention_mask_prefix = attention_mask_prefix
        self._labels_prefix = labels_prefix
        self._loss_prefix = loss_prefix
        self._logits_prefix = logits_prefix

        if bert_kwargs is None:
            bert_kwargs = dict()

        self._encoder = BERT_CLS[task_type].from_pretrained(**bert_kwargs)

    def __call__(self, inputs):
        loss, logits = self._encoder(
            input_ids=inputs[self._input_ids_prefix],
            attention_mask=inputs[self._attention_mask_prefix],
            token_type_ids=None,
            labels=inputs[self._labels_prefix],
            return_dict=False
        )

        inputs[self._loss_prefix] = loss
        inputs[self._logits_prefix] = logits

        return inputs

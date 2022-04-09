from transformers import BertForSequenceClassification

from .base import TorchModel
from .projector import BaseProjector
from .head import BaseHead
from .loss import BaseLoss
from .metric import BaseMetric

BERT_CLS = {
    'classification': BertForSequenceClassification
}


class BertModel(TorchModel, config_name='bert'):
    def __init__(
            self,
            task_type,
            projector,
            head,
            loss,
            metrics,
            bert_kwargs=None
    ):
        super().__init__()

        if bert_kwargs is None:
            bert_kwargs = dict()

        self._projector = projector
        self._encoder = BERT_CLS[task_type].from_pretrained(**bert_kwargs)
        self._head = head
        self._loss = loss
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        projector = BaseProjector.create_from_config(config['projector'])
        head = BaseHead.create_from_config(config['head'])
        loss = BaseLoss.create_from_config(config['loss'])
        metrics = BaseMetric.create_from_config(config['metrics'])

        return BertModel(
            task_type=config['task_type'],
            projector=projector,
            head=head,
            loss=loss,
            metrics=metrics,
            bert_kwargs=config.get('bert_kwargs', None)
        )

    def forward(self, inputs):
        inputs = self._projector(inputs)

        # TODO convert to flexible encoder
        # inputs = self._encoder()
        # inputs = self._head(inputs)
        # inputs = self._loss(inputs)  Here we do loss.backward()
        # inputs = self._metrics(inputs)

        result = self._encoder(
            input_ids=inputs['input_ids'],  # input_embeds use only
            attention_mask=inputs['attention_mask'],
            token_type_ids=None,
            labels=inputs['label'],
            return_dict=False
        )
        loss = result[0]
        inputs['loss'] = loss.item()

        inputs = self._head(inputs)

        inputs = self._loss(inputs)
        inputs = self._metrics(inputs)

        return inputs

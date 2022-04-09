from modeling.data import BaseDataset, ColaDataset
from modeling.model.loss import BaseLoss, IdentityLoss
from modeling.model.metric import BaseMetric, StaticMetric
from modeling.model import BaseModel, TorchModel, BertModel
# TODO optimizer
from utils import GLOBAL_TENSORBOARD_WRITER, parse_args, create_logger, fix_random_seed

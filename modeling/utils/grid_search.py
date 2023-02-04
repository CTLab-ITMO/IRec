import copy

from itertools import product

from callbacks import BaseCallback


class Params:

    def __init__(self, config, params):
        self._initial_config = copy.deepcopy(config)
        self._initial_params = copy.deepcopy(params)

    def __iter__(self):
        keys = []
        values = []

        all_keys = set(self._initial_config.keys()).union(set(self._initial_params.keys()))

        for field_name in all_keys:
            keys.append(field_name)

            initial_field_value = self._initial_config.get(field_name)
            params_fields_value = self._initial_params.get(field_name)

            if initial_field_value:
                if params_fields_value is None:  # We don't want to iterate through this field
                    values.append([initial_field_value])
                elif isinstance(initial_field_value, dict):  # It is composite param, need to go inside
                    field_variations = list(Params(initial_field_value, params_fields_value))
                    values.append(field_variations)
                else:  # Simple param, can take as it is
                    values.append([initial_field_value] + params_fields_value)
            else:
                values.append(self._initial_params[field_name])

        yield from [dict(zip(keys, p)) for p in product(*values)]

        return StopIteration

    @staticmethod
    def dict_to_str(config):
        str_parts = []
        for key, value in config.items():
            if isinstance(value, dict):
                sub_string = Params.dict_to_str(value).split('_')
                str_parts.append('_'.join(list(map(lambda x: '{}.{}'.format(key, x), sub_string))))
            else:
                str_parts.append('{}={}'.format(key, str(value)))
        return '_'.join(str_parts)


if __name__ == '__main__':
    import json

    config = '''{
        "model": {
            "a": -1,
            "b": -1,
            "c": {
                "ca": -1
            }
        },
        "model_params": {
            "b": [1, 2],
            "d": [1, 2],
            "c": {
                "cb": [1, 2]
            }
        }
    }'''
    config = json.loads(config)

    callback_config = '''
    {
        "type": "metric",
        "on_step": 1,
        "loss_prefix": "loss"
    }'''

    import utils

    callback_config = json.loads(callback_config)
    callback = BaseCallback.create_from_config(
        callback_config,
        model=None,
        dataloader=None,
        optimizer=None
    )

    writer = utils.tensorboards.TensorboardWriter('test1')
    writer.add_scalar('lol/lol', 10, 1)
    writer.add_scalar('lol/lol', 20, 2)
    writer.add_scalar('lol/lol', 30, 3)
    writer.add_scalar('lol/lol', 40, 4)
    writer.flush()
    writer.close()

    # utils.GLOBAL_TENSORBOARD_WRITER = utils.tensorboards.TensorboardWriter('test2')
    # utils.GLOBAL_TENSORBOARD_WRITER.add_scalar('lol/lol', 10, 1)
    # utils.GLOBAL_TENSORBOARD_WRITER.add_scalar('lol/lol', 20, 2)
    # utils.GLOBAL_TENSORBOARD_WRITER.add_scalar('lol/lol', 30, 3)
    # utils.GLOBAL_TENSORBOARD_WRITER.add_scalar('lol/lol', 40, 4)
    # utils.GLOBAL_TENSORBOARD_WRITER.flush()
    # utils.GLOBAL_TENSORBOARD_WRITER.close()

    # for param in Params(config['model'], config['model_params']):
    #     print(param)
    #     print(Params.dict_to_str(param))
    #     print()

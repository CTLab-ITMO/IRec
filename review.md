# Review

## Todos

- level embeddings
- fix trie eval
- sos / bos embedding correct train fix
- positions = positions // self._semantic_id_length или reverse?
как именно учитываем codebook_post & item_pos (тот же порядок или inverted)
- как именно находим ближайшего при пересечении по embedding? (не понял о каком embedding речь)

то есть:

1) items (10)
2) semantic_ids (40) -> (1, 2, 3, 4)
3) predicting 11th item (next 4 semantic ids)
4) if single - ok
5) if several? -> dedup # let length be 5 in rqvae, dedup -> 4 + closest by dist
6) if nothing? take all by longest prefix (closest by L^2 / COS / dot)

encoder -> (b_size x 40 x emb_dim)
target  -> (b_size x 4) [(1, 2, 3, 4); (29, 6, 7, 4); ...]
decoder: (bos, 1, 2, 3) -> (1, 2, 3, 4) # causal mask so (bos -> 1), (bos, 1 -> 2), ...
           \___ learnable embed

## Fixed

- next_item_pred / last_item_pred (какие задачи учим и как именно) # can be both tasks
- предсказываем item = предсказываем 4 semantic id? # yes
- как составить датасет для обучения # (map item seq -> semantic id seq)
- берем правдивые semantic id # yes
- у нас авторегрессионный next item prediction? # no (teacher learning)

- fix dataset (take last max_seq items) (last_item fixed)
- single sample from single user (honest comparison)
- correct logits indexing with tgt_mask? (upper remark fixes)

- posterior collapse (как будто все сваливается в один индекс в кодбуке) (fixed eval code)
- в Amazon датасете пофиг на rating? получается учитываются только implicit действия? # байтовый датасет (любое взаимодействие)
- TODO какой базовый класс использовать для seq2seq модели? (LastPred?) # use encoder from SequentialTorchModel
- TODO имя для модели (tiger) # tmp

## Remarks

- обязательно использование reinit unused clusters! (mark)

## Links

- [dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

## Remarks

- no biases on leave one out strategy (обрезаем по строго временному порогу)

## Todo

### Train full encoder-decoder

- На чем обучать? То есть на каких данных запускать backward pass?
- train model

### Collisions

- fix collisisons
- last index = `KMeans(last residuals, n=|last codebook|)` - collision
- remainder = last embedding
- auto increment last index (check paper)
- Research last index aggregation
  
#### possible collisions example

- item1: 1 2 3 0
- item2: 1 2 3 1
- item3: 4 5 6 0/2
- item4: 4 5 6 1/3

### Retreive

- single item -> ok
- too many items -> get embeddings -> score. Softmax(collisions), torch.logsoftmax(logits) -> score -> argmax

### Framework

- positional emb for item & codebook
- splitting item ?

### positional embeddings example

- (000 111 222) - item
- (012 012 012) - codebook

### Fixes in framework

- user_id & codebook_ids -> repr ???
- add last 'sequence' prediction, now only last item is supported
- dataloader (semantic ids lens)

from collections import defaultdict


def dedup(data):
    count_dict = {}

    result = []
    for item in data:
        code = item[2]
        if code not in count_dict:
            count_dict[code] = 0
        unique_index = count_dict[code]
        count_dict[code] += 1
        new_last_element = (*code, unique_index)
        result.append((item[0], item[1], new_last_element))

    return result

def dedup_semantic_ids(semantic_ids): # TODOPK
    result = []
    count_dict = defaultdict(int)
    for semantic_id in semantic_ids:
        unique_index = count_dict[semantic_id]
        count_dict[semantic_id] += 1
        new_last_element = (*semantic_id, unique_index)
        result.append(new_last_element)
    return result

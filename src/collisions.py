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

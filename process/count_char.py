kv_dict = {}
with open('../label.csv') as f:
    for line in f:
        line = line.strip().split(',')
        key = line[0]
        if key == 'name':
            continue
        value = line[1]
        for i in value:
            kv_dict.setdefault(i, 0)
print kv_dict
print len(kv_dict)

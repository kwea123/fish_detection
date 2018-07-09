def read_class_dict(path):
    class_dict = {}
    with open(path) as f:
        for line in f:
            cls_id = line.split(',')[0]
            cls = line.split(',')[1][:-1]
            class_dict[cls_id] = cls
    return class_dict

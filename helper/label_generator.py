import pandas as pd

def csv_to_label_map(path, data_type):
    df = pd.read_csv(path)
    ignore_list = ['kirtipur tower', 'king statue', 'kotilingeshvara', 'kritipur tower', 'til mahadev narayan temple']
    class_list = list(df['class'])
    final_class_list = list()
    for classname in class_list:
        if classname not in ignore_list and classname != 'degutale temple':
            final_class_list.append(classname)
    class_index = 1
    res_dict = {}
    for class_label in final_class_list:
        res_dict[class_label] = class_index
        class_index += 1

    if data_type == 'list-type':
        return final_class_list
    elif data_type == 'dict-type':
        return res_dict
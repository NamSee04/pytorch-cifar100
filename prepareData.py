import os
import pandas as pd


train_file = './data/train.csv'
test_file = './data/test.csv'
val_file = './data/val.csv'

base_path = './data'

if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file)):
    train = []
    val = []
    test = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for label_folder in os.listdir(folder_path):
                label_folder_path = os.path.join(folder_path, label_folder)
                if os.path.isdir(label_folder_path):
                    for file_name in os.listdir(label_folder_path):
                        file_path = os.path.join(label_folder_path, file_name)
                        if os.path.isfile(file_path):
                            if folder == '952_test':
                                test.append({'label': label_folder, 'image_link': file_path})
                            elif folder == '952_train':
                                train.append({'label': label_folder, 'image_link': file_path})
                            else:
                                val.append({'label': label_folder, 'image_link': file_path})

    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)
    df_val = pd.DataFrame(val)

    df_train.to_csv('./data/train.csv')
    df_test.to_csv('./data/test.csv')
    df_val.to_csv('./data/val.csv')
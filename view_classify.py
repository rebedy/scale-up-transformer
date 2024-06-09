import os
import pandas as pd

all_classifier = pd.read_csv('/home/dylee/__workspace/scale-up-transformer/metadata/mimiccxr_all_classifier.csv')
df = pd.DataFrame(all_classifier)
df.drop(df.columns[2:], axis=1, inplace=True)
df.rename(columns={df.columns[0]: "name", df.columns[1]: "view"}, inplace=True)
# print(df.head)


def f(x):
    return x['name'].split('/')[-1]


def f2(x):
    return x['new_name_jpg'].split('.')[0]


df['new_name_jpg'] = df.apply(f, axis=1)
df['new_name'] = df.apply(f2, axis=1)
df.drop(['new_name_jpg'], axis=1, inplace=True)


black_img_list = os.listdir(
    '/home/edlab/dylee/scaleup_transformer/unified_Performers/unified_sut_conditioned_causal_2of2_20220326_10h44m_d256_l4_h4_conditioned_cuda_res512_trans_gen/decoded_images/black_image')
# file = open('/home/edlab/dylee/scaleup_transformer/unified_Performers/black_images.csv', 'w', newline='')
# wr = csv.writer(file)
# view =['PA','AP','LL','LATERAL']
view = [0, 0, 0, 0]
for i in black_img_list:
    name = i.split('.')[0]
    name = name.split('_')[3]
    condition = df.new_name == name
    tmp = df[condition]
    # print("tmp:", tmp['view'])
    if tmp['view'].item() == 'PA':
        view[0] += 1
    elif tmp['view'].item() == 'AP':
        view[1] += 1
    elif tmp['view'].item() == 'LL':
        view[2] += 1
    else:
        view[3] += 1
    # wr.writerow([tmp])

# file.close()
print(view)

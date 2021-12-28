from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import imghdr
import numpy as np
import h5py
from tqdm import tqdm
import glob
import _pickle as cPickle

from medvill.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

from collections import defaultdict
import csv
import albumentations
import albumentations.pytorch

def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b

def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def pre_processing(tokenizer, sentence):
    sentence = sentence.lower()
    if "? -yes/no" in sentence:
        sentence = sentence.replace("? -yes/no", "")
    if "? -open" in sentence:
        sentence = sentence.replace("? -open", "")
    if "? - open" in sentence:
        sentence = sentence.replace("? - open", "")
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
    token =  tokenizer.tokenize(sentence)
    return token
    
def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name') 
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'   : data['image_name'],
        'image'        : img,
        'question'     : data['question'],
        'answer'       : answer,
        'answer_type'  : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type'  : data['phrase_type'],
        'image_organ'  : data['image_organ']}
    return entry


def _load_dataset(args, dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])   
    entries = []
    for sample, answer in zip(samples, answers):
        img_id = sample['image_name']
        if args.vqa_rad == 'all':
            entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'chest':
            if sample['image_organ'] in {'CHEST', ' CHEST', 'CHEST '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'head':
            if sample['image_organ'] in {'HEAD', ' HEAD', 'HEAD '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))
        elif args.vqa_rad == 'abd':
            if sample['image_organ'] in {'ABD', ' ABD', 'ABD '}: entries.append(_create_entry(img_id2val[img_id], sample, answer))
    return entries


class Img2txtDataset(torch.utils.data.Dataset):
    """ Load image-sentence pairs """
    def __init__(self, 
                 args, 
                 data_set, 
                 file_src, 
                 image_root, 
                 img_root_dir,
                 text_root_dir,
                 split, 
                 batch_size, 
                 tokenizer, 
                 max_len, 
                 file_valid_jpgs, 
                 use_num_imgs=-1, 
                 short_sampling_prob=0.1, 
                 sent_reverse_order=False, 
                 bi_uni_pipeline=[],
                 s2s_prob=0, 
                 bi_prob=1, 
                 
                 tasks='report_generation',
                 
                 target_count = 2,
                 max_img_num=1,
                 max_text_len=256,
                 use_first_img=True
                 ):
        
        super().__init__()
        self.data_set = data_set
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        print(' seq2seq {} vs bidirectional {}'.format(self.s2s_prob, self.bi_prob))
        assert(self.s2s_prob + self.bi_prob == 1)

        # read the file into memory
        self.ex_list = []

        if tasks == 'report_generation':
            counter = 0
            
            self.max_img_num = max_img_num
            self.max_text_len = max_text_len
            self.use_first_img = use_first_img
            self.img_root_dir = img_root_dir
            self.text_root_dir = text_root_dir

            self.tokenizer = tokenizer
            
            self.dict_by_studyid = defaultdict(list)
            f = open(file_src, 'r')
            rdr = csv.reader(f)
            for i, line in enumerate(tqdm(rdr)):
                dicom_id, subject_id, study_id, ViewPosition, count = line
                if (int(count) == int(target_count)):
                    self.dict_by_studyid[study_id].append(line)  # {study_id: [[dicom_id, subject_id, study_id, ViewPosition, count],[...],...]}
            self.key_list = list(self.dict_by_studyid.keys())
            print("\nnumber of target subject:", len(self.key_list))
    
            for i, src in enumerate(tqdm(self.key_list)):
                study_id = self.key_list[i]
                count = len(self.dict_by_studyid[study_id])
                if count > self.max_img_num:
                    imgs_meta = random.sample(self.dict_by_studyid[study_id], self.max_img_num)
                    count = self.max_img_num
                else:
                    imgs_meta =self.dict_by_studyid[study_id]
                
                if self.use_first_img == True:
                    imgs_meta = [self.dict_by_studyid[study_id][0]]  # 덮어씌우기
                    count = 1
                
                # image
                img_paths = ''
                for i in range(count):
                    dicom_id, subject_id, studyid, ViewPosition, _ = imgs_meta[i]
                    img_path = os.path.join(self.img_root_dir, 'p'+subject_id[:2], 'p'+subject_id, 's'+studyid, dicom_id+'.jpg')
                    # img_paths += (img_path + '|')
                    img_paths += img_path
                # text
                text_path = os.path.join(self.text_root_dir, 's'+study_id+'.txt')
                with open(text_path, 'r') as f:
                    data = f.read()
                src = data.replace('  ', ' ').replace('  ', ' ').lower()   # Note: 토크나이저가 lower text에서 학습됐음
                src = self.tokenizer.encode(src).ids  # == ids_list # len: max_text_len
                # text_output = torch.tensor(ids_list)  # tensor[max_text_len]
                
                self.ex_list.append((img_paths, src, 1, {'answer_type': ['dummy']}, {'image_organ': ['dummy']}))              
                # self.ex_list.append((img_paths, tokenizer.tokenize(src), 1, {'answer_type': ['dummy']}, {'image_organ': ['dummy']}))              
                counter += 1 
                # if counter == 1000:
                #     break  
                
                
            # if self.data_set == 'valid':
            #     img_dat = [json.loads(l) for l in open(file_valid_jpgs)]
            #     print('Loading {0} valid JPG IDs!'.format(len(img_dat)))
            
            # else: 
            #     img_dat = [json.loads(l) for l in open(file_src)]
            #     print('Loading {0} train JPG IDs!'.format(len(img_dat))) 

            # for idx, src in enumerate(tqdm(img_dat)): # load each img path & txt
            #     src_tk = src['img']
            #     tgt_tk = src['text']
            #     tgt_label = src['label']
            #     if tgt_label == []:
            #         tgt_label = 'Others'
            #     else: pass                
                # self.ex_list.append((d_img, tokenizer.tokenize(d_txt), 1, {'answer_type': ['dummy']}, {'image_organ': ['dummy']}))              
                # counter += 1        
        else: 
            self.ex_list = None
        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        
        instance = self.ex_list[idx]
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)




class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, args, max_pred, mask_prob, vocab_words, indexer, max_len, bar, block_mask=False, new_segment_ids=False, truncate_config={}, mode=None, len_vis_input=None, local_rank=-1, load_vqa_set=False):
        super().__init__()
        self.tasks = args.tasks
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  #tokenizer # function from token to token index
        self.max_len = max_len
        self.bar = bar
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        assert mode in ("s2s", "bi", "bar")
        self.mode = mode

        if mode == 's2s': 
            self.task_idx = 3   # relax projection layer for different tasks
        elif mode == 'bi': 
            self.task_idx = 0
        self.len_vis_input = len_vis_input

        # for images
        self.Resize = transforms.Resize(224)
        self.gray_scale_3ch = transforms.Grayscale(num_output_channels=3)
        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.ans_proc = None
        self.load_vqa_set = load_vqa_set
        
        self.img_reso = 512        # eg. 256 or 384 in my case
        self.rescaler = albumentations.SmallestMaxSize(max_size = self.img_reso)    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
            #self.totensor,
            ])
        
    def __call__(self, instance):
        img_path, tokens_b, target, ans_type, organ = instance

        tokens_a = ['[UNK]'] * self.len_vis_input
        # itm dataset

        truncate_tokens_pair(tokens_a, tokens_b,
            self.len_vis_input + self.max_len_b, max_len_b=self.max_len_b,
            trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
            
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == 's2s':
                segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
            elif self.mode == 'bi':
                segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)
        else:
            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b) # txt token length
        n_pred = min(self.max_pred, max(1, int(round(effective_length * self.mask_prob)))) # masking 갯수
        
        if self.tasks == 'report_generation':
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            

            for i, tk in enumerate(tokens):
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)

            shuffle(cand_pos)
            if random.random() > 0.5:   #Last SEP token 50% mask prob.
                masked_pos = cand_pos[:n_pred-1]
                masked_pos.append(len(tokens)-1)
            else:
                masked_pos = cand_pos[:n_pred]
                
            masked_tokens = [tokens[pos] for pos in masked_pos]

            for pos in masked_pos:
                tokens[pos] = '[MASK]'
        else:
            n_pred = 0
            masked_pos = []
            masked_tokens = []
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3

        if self.bar:
            input_mask[:, :len(tokens_a)+2].fill_(1)
            input_mask[:len(tokens_a)+2, :].fill_(1)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])

        elif self.bar == False:
            if self.mode == "s2s":
                input_mask[:, :len(tokens_a)+2].fill_(1)
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])
            
            elif self.mode == "bi":
                input_mask = torch.tensor([1] * len(tokens) + [0] * n_pad, dtype=torch.long) \
                    .unsqueeze(0).expand(self.max_len, self.max_len).clone()
        
        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        # loading images
        img = Image.open(img_path)     #(2544, 3056) (1840, 2022)
        img = self.gray_scale_3ch(img) #
        img = np.array(img).astype(np.uint8)         # value 0 ~ 255 # (3056, 2544, 3)
        img = self.preprocessor(image=img)["image"]  # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        # if self.len_vis_input < 100:
        #     img = self.Resize(img)
        # else: pass
        img = self.ToTensor(img)
        img = self.res_Normalize(img)   #[3, 3056, 2544]
        vis_pe = torch.arange(2048, dtype=torch.float)
        vis_pe = vis_pe.unsqueeze(0).expand(len(tokens_a), 2048)

        if self.load_vqa_set:
            ans_tk = target
            if ans_type in {"CLOSED", "CLOSED "}:
                ans_type = torch.tensor(0)
            elif ans_type in {"OPEN", "OPEN "}:
                ans_type = torch.tensor(1)
            
            if organ == "CHEST":
                organ = torch.tensor(0)
            elif organ == "HEAD":
                organ = torch.tensor(1)
            elif organ == "ABD":
                organ = torch.tensor(2)

        else:
            ans_tk = torch.tensor(0)
            ans_type = torch.tensor(0)
            organ = torch.tensor(0)
        
        
        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, self.task_idx, img, vis_pe, ans_tk, ans_type, organ)


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, tokenizer, max_len, max_txt_length, new_segment_ids=False, mode="s2s", len_vis_input=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        self.mode = mode
        if self.mode != "s2s":
            raise ValueError("Invalid mode for seq2seq decode: %s" % self.mode)
        self.max_txt_length = max_txt_length
        self.len_vis_input = len_vis_input
        self.Resize = transforms.Resize(224)
        self.gray_scale_3ch = transforms.Grayscale(num_output_channels=3)

        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        
        # for images        
        self.img_reso = 512        # eg. 256 or 384 in my case
        self.rescaler = albumentations.SmallestMaxSize(max_size = self.img_reso)    # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
            #self.totensor,
            ])
        
        
        

    def __call__(self, instance):
        img_path, max_a_len, original_text = instance[:3]        
        tokens_a = ['[UNK]'] * self.len_vis_input

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']

        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_txt_length +
                               max_a_len + 2, self.max_len)

        tokens = padded_tokens_a
        if self.new_segment_ids:
            segment_ids = [4]*(len(padded_tokens_a)) \
                + [5]*(max_len_in_batch - len(padded_tokens_a))
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)
        
        # Token Indexing        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        gt_token = self.tokenizer.tokenize(original_text)
        gt_token_id = self.tokenizer.convert_tokens_to_ids(gt_token)
        
        # input_ids = self.tokenizer.encode(tokens)
        # gt_token = self.tokenizer.encode(original_text)
        # gt_token_id = self.tokenizer.encode(gt_token)

        while True:
            if len(gt_token_id)  <= self.max_txt_length:
                break
            else:
                gt_token_id.pop()
            
        n_pad = self.max_txt_length - len(gt_token_id)
        gt_token_id.extend([0] * n_pad)        
        # assert len(gt_token_id) == 128
        
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)

        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])


        # loading images
        img = Image.open(img_path)
        img = self.gray_scale_3ch(img)
        # img = self.Resize(img)
        # if self.len_vis_input < 100:
        #     img = self.Resize(img)
        #     print("img", img.shape)
        #     exit()
        # else: pass
        img = np.array(img).astype(np.uint8)         # value 0 ~ 255 # (3056, 2544, 3)
        img = self.preprocessor(image=img)["image"]  # albumentations의 output: {'image': numpy.ndarray(256,256,3)}
        img = self.ToTensor(img)
        img = self.res_Normalize(img)
        
        vis_pe = torch.arange(2048, dtype=torch.float)
        vis_pe = vis_pe.unsqueeze(0).expand(len(tokens_a), 2048)

        return (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe, gt_token_id)
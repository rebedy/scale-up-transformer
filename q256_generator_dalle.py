import wandb
# wandb.login()

import os, sys
from time import time, localtime, gmtime, strftime
from collections import OrderedDict
import tqdm, datetime, timeit
import math, csv, json, tempfile
from functools import partial
import numpy as np
from pathlib import Path
import re, logging

import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.empty_cache()
# torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import benchmark
from torch.utils.data import DataLoader, distributed
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.autograd as dist_autograd
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
# from apex.fp16_utils import *
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.autograd.profiler import profile, record_function
# from tokenizers.implementations import ByteLevelBPETokenizer

from utils import *
from helpers import *
from sampling import *
from train_helper import *

from loader import CXRDataset
from transformer_dalle.transformer_i2t import *
import q256_parser_dalle as parser_

MPI_PORT = 11115

##########!##########
start = datetime.datetime.now()
TODAY = str(datetime.date.today().strftime('%Y%m%d'))
NOW = str(start.strftime('_%Hh%Mm'))


##########! Argument Parsing !##########
global args
parser = argparse.ArgumentParser(parents=[parser_.parser], description="sut on transformer(dalle)")

## inference args
                            #!#
parser.add_argument('--infer_batch_size', default=16, type=int)
parser.add_argument('--task', default='val', choices=['val', 'test','train_mini','train'], type=str)

parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--num_beams', default=1, type=int, help="If more than 1, then config beam search.")
parser.add_argument('--do_sample', default=True, type=str2bool, help='If false, generate greedy sampling. Else, the logit filter fuction must be designated.')
parser.add_argument('--filter_logits_fn', default='top_p', choices=['top_k', 'top_p', 'both'], type=str)
parser.add_argument('--top_k', default=100, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--temperature', default=0.7, type=float)

parser.add_argument("--infer_profile", default=True, type=bool)
parser.add_argument('--prefix', default='q256', choices=['q256', 'q512', 'c256', 'c512'], type=str)
parser.add_argument("--infer_save", default="/home/edlab/dylee/scaleup_transformer/logs_infer/", type=str, help="Where to save inference results.")
parser.add_argument("--gen_save", default="/home/dylee/__workspace/scaleup_transformer/infer_log/", type=str, help="Where to save report generation result csv file.")
parser.add_argument("--exp_name", default="NLG_1st_trial", type=str, help="What is this experiment about. Please include number or code at start.")

args = parser.parse_args()

#######!########
                #"/home/edlab/dylee/scaleup_transformer/transformer_q256/"
# args.infer_ckpt= args.save_dir_base+"/sut_q256_20211017_05h14m_totim1_im1_txt256_fixed_bz30_ep50_lr1e-05_relu/epoch=000050.ckpt"
args.infer_ckpt= args.save_dir_base+"/sut_q256_20211019_00h35m_totim1_im1_txt256_fixed_bz16_ep100_lr1e-05_relu/epoch=000009.ckpt"
# args.infer_ckpt= args.save_dir_base+"/sut_q256_20211017_15h28m_totim1_im1_txt256_fixed_bz4_ep50_lr1e-05_relu/epoch=000010.ckpt"



##############!#############
##########! MAIN !##########
##############!#############

#!# Logging setting
random_seed_all( args.seed)   
device = torch.device('cuda')
torch.set_printoptions(precision=10)

if args.infer_profile:
    ori_stdout = sys.stdout
    file_path = os.path.join(args.gen_save, str(args.prefix+"_"+TODAY+NOW+"_"+args.infer_ckpt.split("/")[-1][-10:-5]+"_"+args.filter_logits_fn+"/"))
    os.makedirs(file_path, exist_ok=True)
    log_text = open(os.path.join(file_path, TODAY+NOW+'_inference.log'), 'a')
    sys.stdout = log_text


#!# Dataset settings    
print("\nCXR Dataset Loading...")    
dataset_args={
    "img_root_dir" : args.img_root_dir,
    "text_root_dir" : args.text_root_dir,
    "vqgan_model_path" : args.vqgan_model_path,
    "vqgan_config_path" : args.vqgan_config_path,
    "codebook_indices_path" : args.codebook_indices_path,
    "vocab_file" : args.vocab_file,
    "merge_file" : args.merge_file,
    "target_count" : args.target_count,
    "max_img_num" : args.max_img_num,
    "under_sample" : args.under_sample_type,
    "max_text_len" : args.max_text_len,
}
print("Loading dataset of the task for ", args.task)
if args.task == 'val':
    _ds = CXRDataset(args.val_meta_file, **dataset_args)
elif args.task == 'test':
    _ds = CXRDataset(args.test_meta_file, **dataset_args)
elif args.task == 'train_mini':
    _ds = CXRDataset(args.train_mini_meta_file, **dataset_args)
else:
    _ds = CXRDataset(args.train_meta_file, **dataset_args)
dl = DataLoader(_ds, batch_size=args.infer_batch_size, pin_memory=True, num_workers=args.num_workers)
    
tokenizer = _ds.tokenizer
sos_token_idx = tokenizer.token_to_id("[SOS]")
eos_token_idx = tokenizer.token_to_id("[EOS]")
pad_token_idx = _ds.pad_token_idx

## add args
args.num_tokens = _ds.text_vocab_size  # NOTE: text vocab size
args.num_img_tokens = _ds.img_vocab_size + _ds.max_img_num   # NOTE: img vocab size + num img pad
args.max_seq_len = _ds.img_len * _ds.max_img_num + _ds.max_text_len
args.max_img_num = _ds.max_img_num
args.max_text_len = _ds.max_text_len
args.condition_len = _ds.img_len * _ds.max_img_num
print("condition_len is : ", args.condition_len)
args.img_fmap_size = int(_ds.img_fmap_size)  #16

kargs_transformer_i2t = {
    'dim':args.dim,
    'depth':args.depth,
    'num_tokens' : args.num_tokens,
    'num_img_tokens':args.num_img_tokens,
    'max_seq_len':args.max_seq_len,
    'max_img_num':args.max_img_num,
    'padding_idx' : pad_token_idx,
    'condition_len': args.condition_len,
    'image_fmap_size':args.img_fmap_size,
    'heads':args.heads,
    'dim_head':args.dim_head,
    'reversible':args.reversible,
    'attn_types':args.attn_types,
    'ff_dropout':args.ff_dropout,
    'attn_dropout':args.attn_dropout,
    'stable':args.stable_softmax,
    'shift_tokens':args.shift_tokens,
    'rotary_emb':args.rotary_emb,
    'emb_dropout':args.emb_dropout,
    }

model = TransformerLM_i2t(**kargs_transformer_i2t)
model.cuda()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNumber of params: {n_parameters}")
param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},]

########## ! Load Checkpoint ##########
print("Loading Checkpoint... '{}'".format(args.infer_ckpt))
checkpoint = torch.load(args.infer_ckpt, map_location = lambda storage, loc: storage.cuda(0))
#!# in case we load a DDP model checkpoint to a non-DDP model
model_dict = OrderedDict()
pattern = re.compile('module.')
for k,v in checkpoint['state_dict'].items():
    if re.search("module", k):
        model_dict[re.sub(pattern, '', k)] = v
    else:
        model_dict = checkpoint['state_dict']
model.load_state_dict(model_dict)
# model.load_state_dict(checkpoint['state_dict'])
args.start_epoch = checkpoint['epoch'] + 1
print("     => loaded checkpoint (epoch {})".format(checkpoint['epoch']))
model.eval()



##########! Report Generate !##########
print("Start Generating..")
ep_secs = []
infer_loss, perplexity = [], []
decoded_texts, decoded_texts_csv = {}, []

for epoch in range(args.start_epoch, args.start_epoch+1):
    start = datetime.datetime.now()

    for batch_idx, batch in enumerate(tqdm.tqdm(dl)):
        # for i, batch in enumerate(dl):
        images = batch['images'].cuda()#.to(gpu)#.cuda(args.gpu, non_blocking=True)
        texts = batch['texts'].cuda()#.to(gpu)#.cuda(args.gpu, non_blocking=True)
        
        with torch.no_grad():
            
            #!# compute output loss and ppl
            logits =  model(images, texts)
            target = texts.reshape(-1)
            logit = logits[:, args.condition_len-1:-1].reshape(-1, logits.size(-1))
            
            _loss = F.cross_entropy(logit, target, ignore_index=pad_token_idx)
            _ppl = torch.exp(_loss)
            infer_loss.append(_loss.item())
            perplexity.append(_ppl.item())
            
            #!# text generation
            b, image_seq_len, device = *images.shape, images.device
            start_tok = torch.tensor([[sos_token_idx]]*b).to(device)
            out = torch.cat((images, start_tok), -1)
            
            if args.filter_logits_fn == 'top_k':
                filter_logits_fn = top_k_sampling
                filter_thres = args.top_k
            elif args.filter_logits_fn == 'top_p':
                filter_logits_fn = top_p_sampling
                filter_thres = args.top_p
            elif args.filter_logits_fn == 'both':
                filter_logits_fn = top_k_top_p_filtering
                filter_thres = None
            else:
                raise ValueError('filter_logits_fn must be in (top_k, top_p)')


            for cur_len in range(image_seq_len+1,  args.max_seq_len):
    
                image, text = out[:, :image_seq_len], out[:, image_seq_len:]

                logits = model(image, text)[:, -1, :] # -> logits: [b, num_text_tokens]
                filtered_logits = filter_logits_fn(logits, thres = filter_thres)
                probs = F.softmax(filtered_logits / args.temperature, dim = -1) # [b, num_text_tokens]
                sample = torch.multinomial(probs, 1) # [B, 1]
    
                out = torch.cat((out, sample), dim=-1)

            # break check
            if ( (out[:, image_seq_len:] == eos_token_idx).sum(dim=-1) > 0 ).sum() == b:
                break

            gen_out = out[:, image_seq_len:]  # [b, <text_seq_len]
            ## postprocess
            indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in gen_out]
            for row, idx in enumerate(indices):
                if idx >= 0:
                    gen_out[row, idx+1:] = pad_token_idx             
         
        gt_texts = texts.cuda().tolist()
        gen_out = gen_out.cuda().tolist()

        for img_path_i, study_id_i, gt_text_i, gen_text_i in zip(batch['img_paths'], batch['study_id'], gt_texts, gen_out):
            gt_decoded = tokenizer.decode(gt_text_i, skip_special_tokens=True)
            gen_decoded = tokenizer.decode(gen_text_i, skip_special_tokens=True)
            # print(gen_decoded)
            
            ##to csv
            decoded_texts_csv.append([str(gen_decoded), str(gt_decoded), img_path_i, study_id_i])
            output = {
                'gen_text': gen_decoded,
                'GT_text': gt_decoded,
                'images': img_path_i,
                'study_id':study_id_i,
            }
            decoded_texts[study_id_i] = output

    ## Loss and PPL
    avg_test_loss = np.mean(infer_loss)
    avg_perplexity = np.mean(perplexity)
    print("\navg_test_loss : ", avg_test_loss.item())
    print("avg_perplexity : ", avg_perplexity.item())

    log_loss = {
        'avg_test_loss':avg_test_loss,
        'avg_perplexity': avg_perplexity,
        }

    print("\n     @ The generation took for ", datetime.datetime.now() -start, "\n\n")




    #################!#################
    ##########! MASUREMENTS !##########
    #################!#################

    ## To calculate bleu score
    ''' 
        wandb:       BLEU-1 0.24572
        wandb:       BLEU-2 0.13703
        wandb:       BLEU-3 0.08061
        wandb:       BLEU-4 0.05112
        Cumulative 1-gram  : 0.262
        Cumulative 2-gram  : 0.151
        Cumulative 3-gram  : 0.085
        Cumulative 4-gram  : 0.048
    '''
    # tokenizer_ = ByteLevelBPETokenizer(
    #     args.vocab_file,
    #     args.merge_file,
    # )
    from scipy.stats import gmean
    from rouge import Rouge
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge as cocoRouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    references, candidates = [],[]
    prime, generated = {}, {}
    refs, hypos = {}, {}
    meteor_compute =0
    cnt = 0
    for output in decoded_texts.values():
        # print(type(output['GT_text'])) #'str'
        reference = tokenizer.encode(output['GT_text']).ids
        candidate = tokenizer.encode(output['gen_text']).ids
        references.append([reference])
        candidates.append(candidate)
        
        ##to Rouge
        prime[output['study_id']] = output['GT_text']
        generated[output['study_id']] = output['gen_text']
        
        ##to CiDEr (CoCo)
        refs[output['study_id']] = [str(reference)]
        hypos[output['study_id']] = [str(candidate)]
        
        ## METEOR
        meteor_compute += meteor_score(gt_decoded, gen_decoded)
        
        
    ####  Metrics for Report Generation
    print("\nBLEU encoded")
    try:
        bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
        print(f'Cumulative 1-gram: {bleu1:.3f}')
        print(f'Cumulative 2-gram: {bleu2:.3f}')
        print(f'Cumulative 3-gram: {bleu3:.3f}')
        print(f'Cumulative 4-gram: {bleu4:.3f}')

        print("\nGeometric Mean N-Gram Score (Cumulative Score)")
        bleu_cum1 = bleu4
        bleu_cum2 = gmean([bleu1, bleu2, bleu3, bleu4])
        bleu_cum3 = (bleu1 * bleu2 * bleu3 * bleu4)**(1/4)
        print(f'4-Gram Cumulative BLEU (nltk) : {bleu_cum1:.3f}')
        print(f'4-Gram Cumulative BLEU (scipy): {bleu_cum2:.3f}')
        print(f'4-Gram Cumulative BLEU (hand) : {bleu_cum3:.3f}')
        print("################################################")

        print("\n BLEU encoded With Smoothing.")
        chencherries = [SmoothingFunction().method1, SmoothingFunction().method2, SmoothingFunction().method4, SmoothingFunction().method5, SmoothingFunction().method7]
        #SmoothingFunction().method6,  -> AssertionError: assert p_n[2], "This smoothing method requires non-zero precision for bigrams."
        for chencherry in chencherries:
            bleu1_chencherry = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1, 0, 0, 0))
            bleu2_chencherry = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/2, 1/2, 0, 0))
            bleu3_chencherry = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/3, 1/3, 1/3, 0))
            bleu4_chencherry = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/4, 1/4, 1/4, 1/4))
            print("smoothing method (chencherry) ",chencherry)
            print(f'1-Gram BLEU: {bleu1_chencherry:.3f}')
            print(f'2-Gram BLEU: {bleu2_chencherry:.3f}')
            print(f'3-Gram BLEU: {bleu3_chencherry:.3f}')
            print(f'4-Gram BLEU: {bleu4_chencherry:.3f}')
            print("\n")
        
        print("\nGeometric Mean N-Gram Score (Cumulative Score)")
        bleu_cum1_chencherry = bleu4_chencherry
        bleu_cum2_chencherry = gmean([bleu1_chencherry, bleu2_chencherry, bleu3_chencherry, bleu4_chencherry])
        bleu_cum3_chencherry = (bleu1_chencherry * bleu2_chencherry * bleu3_chencherry * bleu4_chencherry)**(1/4)
        print(f'4-Gram Cumulative BLEU (nltk) : {bleu_cum1_chencherry:.3f}')
        print(f'4-Gram Cumulative BLEU (scipy): {bleu_cum2_chencherry:.3f}')
        print(f'4-Gram Cumulative BLEU (hand) : {bleu_cum3_chencherry:.3f}')
        print("################################################")

                
        print("\nMETEOR")
        meteors = meteor_compute/len(decoded_texts_csv)
        print(meteors)
        print("################################################")


        from rouge import Rouge
        rougeRouge = Rouge()
        rouge_scores_avg = rougeRouge.get_scores(hyps=generated, refs=prime, avg=True)
        print("\nRouge average")
        print(rouge_scores_avg)
        # avg {'r': 0.2488000945382391, 'p': 0.34042311478232845, 'f': 0.2455902113245091}
        print("################################################")



        def cocoscore(ref, hypo):
            """
            ref, dictionary of reference sentences (id, sentence)
            hypo, dictionary of hypothesis sentences (id, sentence)
            score, dictionary of scores
            """
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (cocoRouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
            final_scores = {}
            for scorer, method in scorers:
                score, scores = scorer.compute_score( gts=ref, res=hypo)
                if type(score) == list:
                    for m, s in zip(method, score):
                        final_scores[m] = s
                else:
                    final_scores[method] = score
            return final_scores 


        final_scores = cocoscore(refs, hypos)
        print("\nCoCo Scores ",final_scores)
        print('\nckpt ',args.infer_ckpt,
                '\nBLEU-1 ', bleu1,
                '\nBLEU-2 ', bleu2,
                '\nBLEU-3 ', bleu3,
                '\nBLEU-4 ', bleu4,
                "\nMETEOR ", meteors,
                "\nROUGE avg. ", rouge_scores_avg,
                "\nCoCO scores ", cocoscore
                )
    except:
        bleu1_chencherry, bleu2_chencherry, bleu3_chencherry, bleu4_chencherry = 0.0,0.0,0.0,0.0
        print("Cannot calculate BLEU values!!!!!!!")
        pass
    print("################################################")

    ## csv save
    csv_save=True
    if csv_save:
        csv_name = f"{args.task}_bleu_{bleu1_chencherry:.3f}_{bleu2_chencherry:.3f}_{bleu3_chencherry:.3f}_{bleu4_chencherry:.3f}_{args.filter_logits_fn}_k{args.top_k}_p{args.top_p}_temp_{args.temperature}.csv"
        file_name = os.path.join(file_path, csv_name)
        
        csv_columns = ["gen_text", "GT_text", "images", "study_id"]
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for data in decoded_texts_csv:
                writer.writerow(data)
                
        # df = pd.DataFrame.from_dict(decoded_texts, orient='index').fillna(0)
        # df.to_csv(file_name[:-4]+"_dict.csv")

        print("\n\nPlease check the csv result from ", file_name)
    else:
        pass
    print("\n     @ The whole process took for ", datetime.datetime.now() -start, "\n\n")

if args.infer_profile:
    sys.stdout = ori_stdout
    log_text.close()


# ! ################################################################################################################################
    # start = datetime.datetime.now()


    # meteor_compute =0
    # from nltk.translate.meteor_score import meteor_score

    # refs, hypos = {}, {}
    # references, candidates = [], []
    # references_list, candidates_list = [], []

    # for i, csv_data in enumerate(decoded_texts_csv):
    #     gen_decoded, gt_decoded, img_path_i, study_id_i = csv_data
        
    #     ## METEOR
    #     meteor_compute += meteor_score(gt_decoded, gen_decoded)
        
    #     ##to list BLEU
    #     ref = gt_decoded.split(' ')
    #     hyp = gen_decoded.split(' ')
    #     ref = [ref] # list of references for 1 sentence. #[[refe]]
    #     references_list.append(ref)
    #     candidates_list.append(hyp)
        
    #     ##to BLEU nltk
    #     reference = tokenizer_.encode(gt_decoded).ids
    #     candidate = tokenizer_.encode(gen_decoded).ids
    #     references.append([reference])
    #     candidates.append(candidate)
        
    #     ##to CiDEr (CoCo) & Rouge
    #     refs[study_id_i] = [str(gt_decoded)]
    #     hypos[study_id_i] = [str(gen_decoded)]
        
    #     #CoCo Scores
    #         # {'Bleu_1': 0.14804509941070515, 'Bleu_2': 0.07689400352378514, 'Bleu_3': 0.044902585872141895, 'Bleu_4': 0.02814162839128885, 
    #         # 'METEOR': 0.07823451223106813, 'ROUGE_L': 0.14965303792143597, 'CIDEr': 0.03375043805633285}
            
    #         # refs[study_id_i] = [str(reference)]
    #         # hypos[study_id_i] = [str(candidate)]
    #         # CoCo Scores
    #         # {'Bleu_1': 0.8700747282607217, 'Bleu_2': 0.8592015347455741, 'Bleu_3': 0.8534387867695169, 'Bleu_4': 0.8498454749524433, 
    #         # 'METEOR': 0.5660863441340123, 'ROUGE_L': 0.8623471467391305, 'CIDEr': 0.43017141565713074}






    # #####  Metrics for Report Generation

    # print("\nMETEOR")
    # meteors = meteor_compute/len(decoded_texts_csv)
    # print(meteors)
    # print("\n################################################\n")

    
    # from rouge import Rouge
    # rouge_scores_avg = Rouge().get_scores(hyps=hypos, refs=refs, avg=True)
    # print("\nRouge average")
    # print(rouge_scores_avg)
    # # avg {'r': 0.2488000945382391, 'p': 0.34042311478232845, 'f': 0.2455902113245091}
    
    # rouge_scores = Rouge().get_scores(hyps=hypos, refs=refs, avg=False)
    # def call_rouge(rouge_score):
    #     rouge_dict = {"p":[],"r":[],"f":[]}
    #     for idx, score_dict in enumerate(rouge_score):
    #         rouge_dict["p"].append(score_dict["p"])
    #         rouge_dict["r"].append(score_dict["r"])
    #         rouge_dict["f"].append(score_dict["f"])
    #     prec_max = max(rouge_dict["p"])
    #     rec_max = max(rouge_dict["r"])
    #     if(prec_max!=0 and rec_max!=0):
    #         beta = 1.2
    #         score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
    #     else:
    #         score = 0.0
    #     return score
    # rouge1, rouge2, rougel = [], [], []
    # for idx, score_dict in enumerate(rouge_scores):
    #     rouge1.append(score_dict["rouge-1"])
    #     rouge2.append(score_dict["rouge-2"])
    #     rougel.append(score_dict["rouge-l"])
    # rouge_1 = call_rouge(rouge1)
    # rouge_2 = call_rouge(rouge2)
    # rouge_l = call_rouge(rougel)
    # print("\nRouge max f-score")
    # print("rouge-1 : ", rouge_1)
    # print("rouge-2 : ", rouge_2)
    # print("rouge-l : ", rouge_l)

    # print("\n################################################\n")

    #  # rouge1_r = scores["rouge-1"]["r"]
    #     # rouge2_r = scores["rouge-2"]["r"]
    #     # rougel_r = scores["rouge-l"]["r"]
    #     # rouge1_f = scores["rouge-1"]["f"]
    #     # rouge2_f = scores["rouge-2"]["f"]
    #     # rougel_f = scores["rouge-l"]["f"]

    #     # from nlgeval import NLGEval
    #     # metrics_dict = NLGEval().compute_metrics(prime, generated)
    #     # print("\nNEGEval")        
    #     # print(metrics_dict)



    # print("\n\n BLEU With List.")
    # bleu_1gram = corpus_bleu(references_list, candidates_list, weights=(1, 0, 0, 0))
    # bleu_2gram = corpus_bleu(references_list, candidates_list, weights=(0.5, 0.5, 0, 0))
    # bleu_3gram = corpus_bleu(references_list, candidates_list, weights=(0.33, 0.33, 0.33, 0))
    # bleu_4gram = corpus_bleu(references_list, candidates_list, weights=(0.25, 0.25, 0.25, 0.25))
    # print("\nN-Gram BLEU Scores")
    # print(f'1-Gram BLEU: {bleu_1gram:.2f}')
    # print(f'2-Gram BLEU: {bleu_2gram:.2f}')
    # print(f'3-Gram BLEU: {bleu_3gram:.2f}')
    # print(f'4-Gram BLEU: {bleu_4gram:.2f}')
    # from scipy.stats import gmean
    # bleu_cum1 = bleu_4gram
    # bleu_cum2 = gmean([bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram])
    # bleu_cum3 = (bleu_1gram * bleu_2gram * bleu_3gram * bleu_4gram)**(1/4)
    # print(f'4-Gram Cumulative BLEU (nltk) : {bleu_cum1:.2f}')
    # print(f'4-Gram Cumulative BLEU (scipy): {bleu_cum2:.2f}')
    # print(f'4-Gram Cumulative BLEU (hand) : {bleu_cum3:.2f}')
    # print("\n################################################\n")

    


    # print("\n\n BLEU encoded")
    # bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    # bleu2 = corpus_bleu(references, candidates, weights=(1/2, 1/2, 0, 0))
    # bleu3 = corpus_bleu(references, candidates, weights=(1/3, 1/3, 1/3, 0))
    # bleu4 = corpus_bleu(references, candidates, weights=(1/4, 1/4, 1/4, 1/4))
    # print(f'Cumulative 1-gram: {bleu1:.3f}')
    # print(f'Cumulative 2-gram: {bleu2:.3f}')
    # print(f'Cumulative 3-gram: {bleu3:.3f}')
    # print(f'Cumulative 4-gram: {bleu4:.3f}')
    # print("\nGeometric Mean N-Gram Score (Cumulative Score)")
    # from scipy.stats import gmean
    # bleu_cum1 = bleu4
    # bleu_cum2 = gmean([bleu1, bleu2, bleu3, bleu4])
    # bleu_cum3 = (bleu1 * bleu2 * bleu3 * bleu4)**(1/4)
    # print(f'4-Gram Cumulative BLEU (nltk) : {bleu_cum1:.2f}')
    # print(f'4-Gram Cumulative BLEU (scipy): {bleu_cum2:.2f}')
    # print(f'4-Gram Cumulative BLEU (hand) : {bleu_cum3:.2f}')
    # print("\n################################################\n")



    # print("\n\n BLEU encoded With Smoothing.")
    # chencherry = SmoothingFunction().method7
    # bleu1 = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1, 0, 0, 0))
    # bleu2 = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/2, 1/2, 0, 0))
    # bleu3 = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/3, 1/3, 1/3, 0))
    # bleu4 = corpus_bleu(references, candidates, smoothing_function=chencherry, weights=(1/4, 1/4, 1/4, 1/4))
    # print("\nN-Gram BLEU Score with chencherry")
    # print(chencherry)
    # print(f'1-Gram BLEU: {bleu1:.2f}')
    # print(f'2-Gram BLEU: {bleu2:.2f}')
    # print(f'3-Gram BLEU: {bleu3:.2f}')
    # print(f'4-Gram BLEU: {bleu4:.2f}')
    # print("\nGeometric Mean N-Gram Score (Cumulative Score)")
    # from scipy.stats import gmean
    # bleu_cum1 = bleu4
    # bleu_cum2 = gmean([bleu1, bleu2, bleu3, bleu4])
    # bleu_cum3 = (bleu1 * bleu2 * bleu3 * bleu4)**(1/4)
    # print(f'4-Gram Cumulative BLEU (nltk) : {bleu_cum1:.2f}')
    # print(f'4-Gram Cumulative BLEU (scipy): {bleu_cum2:.2f}')
    # print(f'4-Gram Cumulative BLEU (hand) : {bleu_cum3:.2f}')
    # print("\n################################################\n")
    
    
    

    # ################################################################################################

    # from pycocoevalcap.bleu.bleu import Bleu
    # from pycocoevalcap.rouge.rouge import Rouge as cocoRouge
    # from pycocoevalcap.cider.cider import Cider
    # from pycocoevalcap.meteor.meteor import Meteor

    # def cocoscore(ref, hypo):
    #     """
    #     ref, dictionary of reference sentences (id, sentence)
    #     hypo, dictionary of hypothesis sentences (id, sentence)
    #     score, dictionary of scores
    #     """
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    #     scorers = [
    #         (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #         (Meteor(),"METEOR"),
    #         (cocoRouge(), "ROUGE_L"),
    #         (Cider(), "CIDEr")
    #     ]
    #     final_scores = {}
    #     for scorer, method in scorers:
    #         score, scores = scorer.compute_score( gts=ref, res=hypo)
    #         if type(score) == list:
    #             for m, s in zip(method, score):
    #                 final_scores[m] = s
    #         else:
    #             final_scores[method] = score
    #     return final_scores 
    # final_scores = cocoscore(refs, hypos)
    # print("\nCoCo Scores")
    # print(final_scores)
    # print("\n################################################\n")

    # print("The generation took for ", datetime.datetime.now() -start, "\n\n")


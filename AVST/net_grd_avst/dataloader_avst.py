import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random

def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]

class AVQA_dataset(Dataset):

    def __init__(self, label, audio_dir, video_res14x14_dir, transform=None, mode_flag='train'):


        samples = json.load(open('./AVST/data/json/avqa-train.json', 'r'))

        # nax =  nne
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['answer'] not in ans_vocab:
                ans_vocab.append(sample['answer'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 15    # question length

        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_res14x14_dir
        self.transform = transform

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 60 * len(video_list)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        name = sample['video_id']
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        audio = audio[::6, :]

        visual_out_res18_path = './AVST/data/feats/res18_14x14'
        visual_posi = np.load(os.path.join(self.video_res14x14_dir, name + '.npy'))  
        
        # visual_posi [60, 512, 14, 14], select 10 frames from one video
        visual_posi = visual_posi[::6, :]
        video_idx=self.video_list.index(name)

        for i in range(visual_posi.shape[0]):
            while(1):
                neg_frame_id = random.randint(0, self.video_len - 1)
                if (int(neg_frame_id/60) != video_idx):
                    break

            neg_video_id = int(neg_frame_id / 60)
            neg_frame_flag = neg_frame_id % 60
            neg_video_name = self.video_list[neg_video_id]
            visual_nega_out_res18=np.load(os.path.join(self.video_res14x14_dir, neg_video_name + '.npy'))

            visual_nega_out_res18 = torch.from_numpy(visual_nega_out_res18)
            visual_nega_clip=visual_nega_out_res18[neg_frame_flag,:,:,:].unsqueeze(0)

            if(i==0):
                visual_nega=visual_nega_clip
            else:
                visual_nega=torch.cat((visual_nega,visual_nega_clip),dim=0)

        # visual nega [60, 512, 14, 14]

        # question
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
            # print(f'mg {pos}') if len(question) > self.max_len else None
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # answer
        answer = sample['answer']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {'audio': audio, 'visual_posi': visual_posi, 'visual_nega': visual_nega, 'question': ques, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):

        audio = sample['audio']
        visual_posi = sample['visual_posi']
        visual_nega = sample['visual_nega']
        label = sample['label']

        return { 
                'audio': torch.from_numpy(audio), 
                'visual_posi': sample['visual_posi'],
                'visual_nega': sample['visual_nega'],
                'question': sample['question'],
                'label': label}
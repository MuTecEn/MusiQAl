from __future__ import print_function
import sys 
sys.path.append("./AVST") 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from net_grd_baseline.dataloader_qa_grd_baseline import *
from net_grd_baseline.nets_qa_grd_baseline import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./MUSIC-AVQA_CVPR2022/runs/AVST/'+TIMESTAMP)

print("\n--------------- MUSIC-AVQA baseline --------------- \n")

def batch_organize(audio_data, posi_img_data, nega_img_data):

    # print("audio data: ", audio_data.shape)
    (B, T, C) = audio_data.size()
    audio_data_batch=audio_data.view(B*T,C)
    batch_audio_data = torch.zeros(audio_data_batch.shape[0] * 2, audio_data_batch.shape[1])


    (B, T, C, H, W) = posi_img_data.size()
    posi_img_data_batch=posi_img_data.view(B*T,C,H,W)
    nega_img_data_batch=nega_img_data.view(B*T,C,H,W)


    batch_image_data = torch.zeros(posi_img_data_batch.shape[0] * 2, posi_img_data_batch.shape[1], posi_img_data_batch.shape[2],posi_img_data_batch.shape[3])
    batch_labels = torch.zeros(audio_data_batch.shape[0] * 2)
    for i in range(audio_data_batch.shape[0]):
        batch_audio_data[i * 2, :] = audio_data_batch[i, :]
        batch_audio_data[i * 2 + 1, :] = audio_data_batch[i, :]
        batch_image_data[i * 2, :] = posi_img_data_batch[i, :]
        batch_image_data[i * 2 + 1, :] = nega_img_data_batch[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return batch_audio_data, batch_image_data, batch_labels

def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    # print(train_loader)
    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

        optimizer.zero_grad()
        out_qa, out_match,match_label = model(audio, visual_posi,visual_nega, question)
        # print(sample)
        # print(f'length of out_qa: {len(out_qa)}, target: {len(target)} ')
        loss_qa = criterion(out_qa, target)
        loss=loss_qa

        writer.add_scalar('data/both',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)
            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()


    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('./AVST/data/json/avqa-test.json', 'r'))

    A_count = []
    A_cmp = []
    A_ext = []
    A_caus = []
    A_temp = []
    V_count = []
    V_loc = []
    V_ext = []
    V_temp = []
    V_caus = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    AV_caus = []
    AV_purp = []

    que_id=[]
    pred_results=[]

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)
            preds=preds_qa
            _, predicted = torch.max(preds.data, 1)
            print(predicted.size())
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            # save pred results
            pred_bool=predicted == target
            for index in range(len(pred_bool)):
                pred_results.append(pred_bool[index].cpu().item())
                que_id.append(question_id[index].item())

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])

            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                    print(A_count)
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
                    print(A_cmp)
                elif type[1] == 'Temporal':
                    A_temp.append((predicted == target).sum().item())
                    print(A_temp)
                elif type[1] == 'Existential':
                    A_ext.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    A_caus.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
                elif type[1] == 'Existential':
                    V_ext.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    V_temp.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    V_caus.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                    # AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())
                elif type[1] == 'Causal':
                    AV_caus.append((predicted == target).sum().item())
                elif type[1] == 'Purpose':
                    AV_purp.append((predicted == target).sum().item())

    with open("./AVST/net_grd_baseline/pred_results/net_grd_baseline.txt", 'w') as f:
        for index in range(len(que_id)):
            f.write(str(que_id[index])+' '+str(pred_results[index]) + '\n')
    
    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Ext Accuracy: %.2f %%' % (
            100 * sum(A_ext) / len(A_ext)))
    print('Audio Temp Accuracy: %.2f %%' % (
            100 * sum(A_temp) / len(A_temp)))
    print('Audio Caus Accuracy: %.2f %%' % (
            100 * sum(A_caus) / len(A_caus)))
    print('Audio Accuracy: %.2f %%' % (
        100 * (sum(A_count) + sum(A_cmp) + sum(A_ext) + sum(A_temp) + sum(A_caus)) / (len(A_count) + len(A_cmp) + len(A_ext) + len(A_temp) + len(A_caus))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Existential Accuracy: %.2f %%' % (
            100 * sum(V_ext) / len(V_ext)))
    print('Visual Temporal Accuracy: %.2f %%' % (
            100 * sum(V_temp) / len(V_temp)))
    print('Visual Causal Accuracy: %.2f %%' % (
            100 * sum(V_caus) / len(V_caus)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc) + sum(V_ext) + sum(V_temp) + sum(V_caus)) / (len(V_count) + len(V_loc) + len(V_ext) + len(V_temp) + len(V_caus))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))
    print('AV Causal Accuracy: %.2f %%' % (
            100 * sum(AV_caus) / len(AV_caus)))
    print('AV Purpose Accuracy: %.2f %%' % (
            100 * sum(AV_purp) / len(AV_purp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp) + sum(AV_caus) + sum(AV_purp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp) + len(AV_caus) + len(AV_purp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of MUSIC Audio-Visual Question Answering')

    parser.add_argument(
        "--audio_dir", type=str, default='./AVST/data/feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='./AVST/data/frames', help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='./AVST/data/feats/res18_14x14', help="video dir")
    
    parser.add_argument(
        "--label_train", type=str, default="./AVST/data/json/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="./AVST/data/json/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="./AVST/data/json/avqa-test.json", help="test csv file")
    parser.add_argument(
        '--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=4, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='./AVST/grounding_gen/models_grounding_gen/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='net_grd_baseline_best', help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0, 1, 2, 3', help='gpu device number')


    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = device
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.manual_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    myis = []
    if args.mode == 'train':
        logger.info("Starting training loop...")
        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        # print(train_dataset.__getitem__(0).keys())
        # print(f"q: {train_dataset.__getitem__(0)['question'].shape[0]}")
        # print(f"l: {train_dataset.__getitem__(0)['label']}")
        # print(f"qID: {train_dataset.__getitem__(0)['question_id']}")
        # for i in range(len(train_dataset)):
        #     sample = train_dataset.__getitem__(i)
        #     sample_info = {
        #         # 'keys': list(sample.keys()),
        #         'question_shape': sample['question'].shape,
        #         'label': sample['label'],
        #         'question_id': sample['question_id']
        #     }
        #     # print( sample['question'].shape[0])
        #     # assert sample['question'].shape[0]==14, f"q: {sample['question_id']}"
        #     myis.append(sample_info)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


        # ===================================== load pretrained model ===============================================
        # None
        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):
            logger.info(f"Epoch {epoch} started")
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader,epoch)
            # if F >= best_F:
            #     best_F = F
            torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

    else:
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        print(test_dataset.__len__())
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        print('Data Loaded!')
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        print('Model Loaded!')
        test(model, test_loader)


if __name__ == '__main__':
    main()
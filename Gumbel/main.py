import argparse
import dataloader
import kbs_model
from tqdm import tqdm
import torch.nn as nn
import torch
import random
import numpy as np
import model_file
import pickle
import torch.nn.functional as F

def get_args():

    arg = argparse.ArgumentParser()

    # train parameter
    arg.add_argument('--batch_size', default=32, type=int)
    arg.add_argument('--lr', default=0.005, type=float)
    arg.add_argument("--dataset", nargs="+", default=['reviews_Automotive_5'])
    arg.add_argument("--data_path", type=str,default='./data/')
    arg.add_argument("--word_embed_file",type=str,default='./data/glove.6B.300d.txt')
    arg.add_argument('--seed', default=2019, type=int)
    arg.add_argument('--word_embedding' , default=300,type=int,help='word_embedding')
    arg.add_argument('--gpu_id', default=0, type=int)

    # NARRE model parameter
    arg.add_argument('--k1',default=80, type=int,help='filter size')
    arg.add_argument('--k2', default=32 , type=int, help='ID embedding dim')
    arg.add_argument('--t', default=32, type=int, help='hidden size of attention network')
    arg.add_argument('--n', default= 20, type=int,help='final dim of representation')
    arg.add_argument('--num_filters', default=100, type=int,help='number of filters per filter size')
    arg.add_argument('--dropout', default=0, type=int)

    # ANR
    arg.add_argument('--K', default=5, type=int, help= 'Number of Aspects')
    arg.add_argument('--h1',default=10,type=int, help='Dimensionality of the Aspect-level Representations')
    arg.add_argument('--h2', default=50, type=int,help='Dimensionality of the Hidden Layers used for Aspect Importance Estimation')

    #D_ATT
    arg.add_argument('--latt', default=200, type=int)
    arg.add_argument('--gatt', default=100, type=int)
    #arg.add_argument('--input_size', default=10000, type=int)

    #NCF
    arg.add_argument("--factor_num", type=int,default=32, help="predictive factors numbers in the model")
    arg.add_argument("--num_layers", type=int,default=3, help="number of layers in MLP model")

    arg.add_argument('--base_model', default='NARRE', type=str,choices=['PMF','NCF','NARRE','D_Att','ANR','NRPA','DeepConn'])
    arg.add_argument('--pdf', default='Gumbel',choices=['Gumbel','GMM','Poisson','Expon','Weibull','Frechet','blank'], type=str,
                     help='the probability density function used on base model')

    arg.add_argument('--epoch', default=10,type=int)
    arg.add_argument('--device', default=0, type=int)

    arg.add_argument('--step_size', default=1, type=int)
    arg.add_argument('--gamma', default=0.8, type=int)

    return arg.parse_args()


if __name__ == '__main__':

    args = get_args()
    data_list = args.dataset
    batch_size = args.batch_size
    learning_rate = args.lr

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(args.seed)
        torch.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f'lr:{args.lr},step:{args.step_size},gamma:{args.gamma}')


    for data_name in data_list:

        print(data_name)
        vocab_dict_path = args.word_embed_file
        file_path = args.data_path + data_name + '.json'
        glove_data = 'data/' +data_name +'_.glove_data.pkl'
        glove_matrix = 'data/' + data_name + '_glove_matrix.pkl'

        glove_data, matrix, review_len = dataloader.word_to_id(glove_data,glove_matrix,vocab_dict_path,file_path)
        train_data,test_data,user_dict,item_dict,u_max,i_max, num_users,num_items = dataloader.prepare_data(glove_data)
        batch = dataloader.Batch(train_data, test_data,user_dict,item_dict, u_max, i_max, batch_size, review_len,train=True) #(review_len是一条评论的长度)

        if args.base_model == 'NARRE':
            mainmodel = kbs_model.NARRE(num_users,num_items,matrix,review_len,args)

        elif args.base_model == 'PMF':
            mainmodel = kbs_model.PMF(num_users,num_items,args)

        elif args.base_model == 'D_Att' :
            mainmodel = kbs_model.D_Att(matrix,num_users,num_items,review_len,args)

        elif args.base_model == 'DeepConn':
            mainmodel = kbs_model.DeepConn(matrix, num_users, num_items, review_len, args)

        elif args.base_model == 'NCF':
            mainmodel = kbs_model.NCF(num_users,num_items,args)

        elif args.base_model == 'NRPA':
            mainmodel = kbs_model.NRPA(num_users,num_items,matrix, review_len, args)

        elif args.base_model == 'ANR':
            mainmodel = kbs_model.ANR(matrix,num_users,num_items, args)

        if use_cuda:
            mainmodel = mainmodel.cuda(args.device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(mainmodel.parameters(), learning_rate, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)
        model = f'{args.base_model}_{args.pdf}'
        print(f'model: {model}')
        best_mse = 999
        improve = 0

        for epoch in range(args.epoch):
            batch.set_train(True)
            mainmodel.train()
            for encode_vectors, user_review_lists, uitem_id_list, item_review_lists,\
                iuser_id_list, overalls, user_id,item_id,u_rating_ratio,i_rating_ratio in batch:

                if use_cuda:
                    user_review_lists = user_review_lists.cuda(args.device)
                    user_id = user_id.cuda(args.device)
                    item_review_lists = item_review_lists.cuda(args.device)
                    item_id = item_id.cuda(args.device)
                    overalls = overalls.cuda(args.device)
                    u_rating_ratio = u_rating_ratio.cuda(args.device)
                    i_rating_ratio = i_rating_ratio.cuda(args.device)
                    uitem_id_list = uitem_id_list.cuda(args.device)
                    iuser_id_list = iuser_id_list.cuda(args.device)


                for p in mainmodel.parameters():
                    p.requires_grad = True

                optimizer.zero_grad()

                pre = mainmodel(user_id,item_id,user_review_lists,item_review_lists,u_rating_ratio,i_rating_ratio,args,uitem_id_list,iuser_id_list)
                loss = F.mse_loss(pre,overalls)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch)

            batch.set_train(False)
            mainmodel.eval()
            total_loss = 0
            num = 0
            for encode_vectors, user_review_lists, uitem_id_list, item_review_lists, iuser_id_list,\
                overalls, user_id, item_id, u_rating_ratio,i_rating_ratio in batch:

                if use_cuda:
                    user_review_lists = user_review_lists.cuda(args.device)
                    user_id = user_id.cuda(args.device)
                    item_review_lists = item_review_lists.cuda(args.device)
                    item_id = item_id.cuda(args.device)
                    overalls = overalls.cuda(args.device)
                    u_rating_ratio = u_rating_ratio.cuda(args.device)
                    i_rating_ratio = i_rating_ratio.cuda(args.device)
                    uitem_id_list = uitem_id_list.cuda(args.device)
                    iuser_id_list = iuser_id_list.cuda(args.device)


                for p in mainmodel.parameters():
                    p.requires_grad = False

                pre = mainmodel(user_id,item_id,user_review_lists,item_review_lists,u_rating_ratio,i_rating_ratio,args,uitem_id_list,iuser_id_list)
                loss = torch.sum(torch.abs(pre-overalls))
                total_loss += loss
                num += len(pre)

            total_loss /= num
            if total_loss < best_mse:
                best_mse = total_loss
                torch.save(mainmodel.state_dict(),f'{model}_params.pkl')
            print(f'epoch {epoch} test loss {total_loss}')

        mainmodel.load_state_dict(torch.load(f'{model}_params.pkl'))
        batch.set_train(False)
        mainmodel.eval()
        real_score = [[],[],[],[],[]]
        best_mae = 0
        num =0
        score_num = [0, 0, 0, 0, 0]
        step = 0
        for encode_vectors, user_review_lists, uitem_id_list, item_review_lists, iuser_id_list, \
            overalls, user_id, item_id, u_rating_ratio, i_rating_ratio in tqdm(batch):

            if use_cuda:
                user_review_lists = user_review_lists.cuda(args.device)
                user_id = user_id.cuda(args.device)
                item_review_lists = item_review_lists.cuda(args.device)
                item_id = item_id.cuda(args.device)
                overalls = overalls.cuda(args.device)
                u_rating_ratio = u_rating_ratio.cuda(args.device)
                i_rating_ratio = i_rating_ratio.cuda(args.device)
                uitem_id_list = uitem_id_list.cuda(args.device)
                iuser_id_list = iuser_id_list.cuda(args.device)

            for p in mainmodel.parameters():
                p.requires_grad = False

            pre = mainmodel(user_id, item_id, user_review_lists, item_review_lists, u_rating_ratio, i_rating_ratio,args, uitem_id_list, iuser_id_list)
            best_mae += torch.sum(torch.abs(pre-overalls))
            num += len(overalls)
            for i in range(1,6):
                mask = overalls==i
                score_num[i-1] += float(torch.sum(mask).cpu())
                a = torch.masked_select(pre, mask.cuda()).cpu().tolist()
                real_score[i-1] += a

        error = {}
        for i in range(1,6):
            score = f'{i}_score'
            a = float(torch.sum(torch.abs(torch.tensor(real_score[i-1]) - float(i)))/len(real_score[i-1]))
            error[score] = format(a, '.3f')

        print(f'{data_name} each rating error {error}')
        print(f'{data_name} best test mae {best_mae/num}')




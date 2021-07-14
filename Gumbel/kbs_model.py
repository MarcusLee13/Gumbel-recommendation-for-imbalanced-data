import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np
import matplotlib


class NARRE(nn.Module):

    def __init__(self,user_num,item_num,word_weight_matrix,review_len,args):
        super(NARRE, self).__init__()

        self.review_embeds = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.review_embeds.weight = nn.Parameter(word_weight_matrix, requires_grad=False)

        self.ureview_conv2d = torch.nn.Conv2d(1, args.num_filters, \
                                            kernel_size=(3,args.word_embedding), padding=(1,0))

        self.ireview_conv2d = torch.nn.Conv2d(1, args.num_filters, \
                                            kernel_size=(3,args.word_embedding), padding=(1,0))

        self.uid_embeds = nn.Embedding(user_num+1, args.k2)
        self.iid_embeds = nn.Embedding(item_num + 1, args.k2)

        self.uidfm_embeds = nn.Embedding(user_num + 1, args.k2)
        self.iidfm_embeds = nn.Embedding(item_num + 1, args.k2)

        self.Wau = nn.Parameter(torch.Tensor(args.num_filters, args.t),requires_grad=True)
        torch.nn.init.uniform_(self.Wau,-0.1,0.1)
        self.Wru =nn.Parameter(torch.Tensor(args.k2, args.t),requires_grad=True)
        torch.nn.init.uniform_(self.Wru,-0.1,0.1)
        self.Wpu = nn.Linear(args.t, args.t)
        torch.nn.init.uniform_(self.Wpu.weight, -0.1, 0.1)
        self.bbu = nn.Parameter(torch.Tensor(1), requires_grad=True)
        torch.nn.init.constant_(self.bbu, 0.1)

        self.Wai = nn.Parameter(torch.Tensor(args.num_filters, args.t), requires_grad=True)
        torch.nn.init.uniform_(self.Wai, -0.1, 0.1)
        self.Wri = nn.Parameter(torch.Tensor(args.k2, args.t), requires_grad=True)
        torch.nn.init.uniform_(self.Wri, -0.1, 0.1)
        self.Wpi = nn.Linear(args.t, args.t,bias=True)
        torch.nn.init.uniform_(self.Wpi.weight, -0.1, 0.1)
        self.bai = nn.Parameter(torch.Tensor(args.t), requires_grad=True)
        torch.nn.init.constant_(self.bai, 0.1)
        self.bbi = nn.Parameter(torch.Tensor(1), requires_grad=True)
        torch.nn.init.constant_(self.bbi, 0.1)

        self.Wu = nn.Linear(args.num_filters,32)
        torch.nn.init.uniform_(self.Wu.weight, -0.1, 0.1)
        self.Wi = nn.Linear(args.num_filters, 32, bias=False)
        torch.nn.init.uniform_(self.Wi.weight, -0.1, 0.1)

        self.Wmul = nn.Linear(32,1,bias=False)
        torch.nn.init.uniform_(self.Wmul.weight, -0.1, 0.1)

        self.u_bias  = nn.Embedding(user_num+1,1)
        torch.nn.init.constant_(self.u_bias.weight, 0.1)
        self.i_bias = nn.Embedding(item_num+1, 1)
        torch.nn.init.constant_(self.i_bias.weight, 0.1)
        self.globalbias = nn.Parameter(torch.Tensor(1),requires_grad=True)
        torch.nn.init.constant_(self.globalbias, 0.1)

        self.linear = nn.Linear(32,5)
        if args.pdf == 'Gumbel':
            self.p = Gumbel(user_num,item_num)
        elif args.pdf == 'Poisson':
            self.p = Poisson(user_num,item_num)
        elif args.pdf == 'GMM':
            self.p = GMM(user_num,item_num)
        elif args.pdf == 'Expon':
            self.p = Expon(user_num,item_num)
        elif args.pdf == 'Weibull':
            self.p = Weibull(user_num,item_num)
        elif args.pdf == 'Frechet':
            self.p = Frechet(user_num,item_num)

    def forward(self, uid,iid,ureview,ireview,u_rating_ratio,i_rating_ratio,args,item_list,user_list):

        uid_feature = self.uid_embeds(user_list)  #(batch,1,k2)
        iid_feature = self.iid_embeds(item_list)
        uidfm = self.uidfm_embeds(uid)
        iidfm = self.iidfm_embeds(iid)

        maxpool_size = ureview.size()[2]
        ureview = self.review_embeds(ureview)
        ureview_fea = ureview.view(-1,1,maxpool_size,args.word_embedding)
        ureview_fea = self.ureview_conv2d(ureview_fea)
        ureview_fea = ureview_fea.view(len(ureview),-1,args.num_filters,maxpool_size,1).squeeze()
        ureview_fea = F.max_pool2d(ureview_fea, (1,maxpool_size)).squeeze() #(32,u_max,100)
        ureview_fea = F.dropout(ureview_fea,args.dropout)
        maxpool_size = ireview.size()[2]
        ireview = self.review_embeds(ireview)
        ireview_fea = ireview.view(-1, 1, maxpool_size, args.word_embedding)
        ireview_fea = self.ireview_conv2d(ireview_fea) #(N,review_len,num_filters)
        ireview_fea = ireview_fea.view(len(ireview), -1, args.num_filters, maxpool_size, 1).squeeze()
        ireview_fea = F.max_pool2d(ireview_fea, (1,maxpool_size)).squeeze()
        ireview_fea = F.dropout(ireview_fea,args.dropout)

        l1 = torch.matmul(ureview_fea,self.Wau) #(n,review_len,att_size)
        l2 =torch.matmul(iid_feature,self.Wru)
        u_j = F.relu(self.Wpu(l1+l2)) + self.bbu
        u_a = F.softmax(u_j,dim = -1) #(N,review_len,10)

        l1 = torch.matmul(ireview_fea, self.Wai)  # (n,review_len,att_size)
        l2 = torch.matmul(uid_feature, self.Wri)
        i_j = F.relu(self.Wpi(l1 + l2)) + self.bbi
        i_a = F.softmax(i_j, dim=-1)  # (N,review_len,10)

        ufea = torch.matmul(u_a.permute(0,2,1) , ureview_fea) #(N,10,num_filters)
        ufea = torch.sum(ufea,dim=1).unsqueeze(1)
        ufea = F.dropout(ufea,args.dropout)
        ufea = self.Wu(ufea) + uidfm #(N,att_size,32)
        ifea = torch.matmul(i_a.permute(0, 2, 1), ireview_fea)  # (N,10,num_filters)
        ifea = torch.sum(ifea, dim=1).unsqueeze(1)
        ifea = F.dropout(ifea)
        ifea = self.Wi(ifea) + iidfm

        if args.pdf == 'blank':
            score = self.Wmul(F.dropout(ufea*ifea,args.dropout))
            score = score + self.u_bias(uid) + self.i_bias(iid) + self.globalbias

        else:
            score = self.linear(ufea * ifea)
            score = self.p(score, uid, iid, u_rating_ratio, i_rating_ratio)

        return score.squeeze()

class PMF(nn.Module):

    def __init__(self,num_user,num_item,args):
        super(PMF, self).__init__()

        self.u = nn.Embedding(num_user + 1, 20)
        torch.nn.init.constant_(self.u.weight,0.5)
        self.i = nn.Embedding(num_item + 1,20)
        torch.nn.init.constant_(self.i.weight,0.5)

        self.globalOffset = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        self.uid_userOffset = nn.Embedding(num_user + 1, 1)
        torch.nn.init.uniform_(self.uid_userOffset.weight, -1, 1)
        self.uid_userOffset.weight.requires_grad = True
        self.iid_itemOffset = nn.Embedding(num_item + 1, 1)
        torch.nn.init.uniform_(self.iid_itemOffset.weight, -1, 1)
        self.iid_itemOffset.weight.requires_grad = True

        self.linear = nn.Linear(20, 5)
        if args.pdf == 'Gumbel':
            self.p = Gumbel(num_user, num_item)
        elif args.pdf == 'GMM':
            self.p = GMM(num_user, num_item)
        elif args.pdf == 'Poisson':
            self.p = Poisson(num_user, num_item)
        elif args.pdf == 'Expon':
            self.p = Expon(num_user, num_item)
        elif args.pdf == 'Weibull':
            self.p = Weibull(num_user, num_item)
        elif args.pdf == 'Frechet':
            self.p = Frechet(num_user, num_item)


    def forward(self,uid,iid,user_review,item_review,u_rating_ratio,i_rating_ratio,args,a,b):

        u = self.u(uid)
        i = self.i(iid)

        if args.pdf == 'blank':
            rate = torch.matmul(u,i.permute(0,2,1)) + self.uid_userOffset(uid) + self.iid_itemOffset(iid) + self.globalOffset

        else:
            rate = u * i
            rate = self.linear(rate)
            rate = self.p(rate, uid, iid, u_rating_ratio, i_rating_ratio)

        return rate.squeeze()


class ANR_ARL(nn.Module):

    def __init__(self,args):
        super(ANR_ARL, self).__init__()

        # Aspect Embeddings
        self.aspEmbed = nn.Embedding(args.K, 3*args.h1)
        self.aspEmbed.weight.requires_grad = True

        # Aspect-Specific Projection Matrices
        self.aspProj = nn.Parameter(torch.Tensor(args.K, args.word_embedding, args.h1),
                                    requires_grad=True)
        self.aspEmbed.weight.data.uniform_(-0.01, 0.01)
        self.aspProj.data.uniform_(-0.01, 0.01)

    def forward(self, review,args):  #batch_docIn:bsz x max_doc_len x word_embed_dim

        # Loop over all aspects
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        for a in range(5):

            # Aspect-Specific Projection of Input Word Embeddings
            batch_aspProjDoc = torch.matmul(review, self.aspProj[a]) #(bsz x max_doc_len x h1)

            # Aspect Embedding: (bsz x h1 x 1) after tranposing!
            bsz = review.size()[0]
            batch_aspEmbed = self.aspEmbed(torch.LongTensor(bsz, 1).fill_(a).cuda())
            batch_aspEmbed = torch.transpose(batch_aspEmbed, 1, 2)  #(bsz x h1 x 1)

            # Context-based Word Importance
            batch_aspProjDoc_padded = F.pad(batch_aspProjDoc,(0,0,1,1),'constant',0)
            batch_aspProjDoc_padded =  batch_aspProjDoc_padded.unfold(1,3,1)
            batch_aspProjDoc_padded = batch_aspProjDoc_padded.reshape(len(batch_aspProjDoc_padded),-1,3*10)
            batch_aspAttn = torch.matmul(batch_aspProjDoc_padded, batch_aspEmbed) #(bsz x max_doc_len x 1)
            batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

            # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
            # (bsz x max_doc_len x h1) and (bsz x max_doc_len x 1) -> (bsz x h1)
            batch_aspRep = batch_aspProjDoc * batch_aspAttn.expand_as(batch_aspProjDoc)
            batch_aspRep = torch.sum(batch_aspRep,dim=1) #(bsz x h1)

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2)) #(bsz x 1 x max_doc_len )
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1)) #(bsz x 1 x h1)

        # Reshape the Attentions & Representations
        # batch_aspAttn:	(bsz x num_aspects x max_doc_len)
        # batch_aspRep:		(bsz x num_aspects x h1)
        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)

        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep


'''
Aspect Importance Estimation (AIE)
'''
class ANR_AIE(nn.Module):

    def __init__(self, args):
        super(ANR_AIE, self).__init__()

        # Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations
		# This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
        self.W_a = nn.Parameter(torch.Tensor(args.h1, args.h1), requires_grad = True)

        # User "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_u = nn.Parameter(torch.Tensor(args.h2, args.h1), requires_grad = True)
        self.w_hu = nn.Parameter(torch.Tensor(args.h2, 1), requires_grad = True)

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = nn.Parameter(torch.Tensor(args.h2, args.h1), requires_grad = True)
        self.w_hi = nn.Parameter(torch.Tensor(args.h2, 1), requires_grad = True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.W_a.data.uniform_(-0.01, 0.01)

        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)

        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    '''
	[Input]		userAspRep:		bsz x num_aspects x h1
	[Input]		itemAspRep:		bsz x num_aspects x h1
	'''

    def forward(self,userAspRep, itemAspRep):

        userAspRepTrans = torch.transpose(userAspRep, 1, 2)
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2) #bsz x h1 x num_aspects

        affinityMatrix = torch.matmul(userAspRep, self.W_a) #bsz x num_aspects x h1
        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans) #bsz x num_aspects x num_aspects
        affinityMatrix = F.relu(affinityMatrix)

        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)
        H_u_2 = torch.matmul(H_u_2, torch.transpose(affinityMatrix, 1, 2))
        H_u = F.relu(H_u_1 + H_u_2)

        # User Aspect-level Importance
        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)

        # User Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        userAspImpt = torch.transpose(userAspImpt, 1, 2)
        userAspImpt = F.softmax(userAspImpt, dim = 1)
        userAspImpt = torch.squeeze(userAspImpt, 2)

        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)
        H_i_2 = torch.matmul(H_i_2, affinityMatrix)
        H_i = F.relu(H_i_1 + H_i_2)

        # Item Aspect-level Importance
        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)

        # Item Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)
        itemAspImpt = F.softmax(itemAspImpt, dim = 1)
        itemAspImpt = torch.squeeze(itemAspImpt, 2)

        return userAspImpt, itemAspImpt


class ANR_RatingPred(nn.Module):

    def __init__(self, num_user, num_item, args):

        super(ANR_RatingPred, self).__init__()

        # Global Offset/Bias (Trainable)
        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad = True)

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = nn.Embedding(num_user+1, 1)
        self.uid_userOffset.weight.requires_grad = True

        self.iid_itemOffset = nn.Embedding(num_item+1, 1)
        self.iid_itemOffset.weight.requires_grad = True

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

        self.linear = nn.Linear(15,5,bias=False)
        self.linear = nn.Linear(15, 5)
        if args.pdf == 'Gumbel':
            self.p = Gumbel(num_user, num_item)
        elif args.pdf == 'GMM':
            self.p = GMM(num_user, num_item)
        elif args.pdf == 'Poisson':
            self.p = Poisson(num_user, num_item)
        elif args.pdf == 'Expon':
            self.p = Expon(num_user, num_item)
        elif args.pdf == 'Weibull':
            self.p = Weibull(num_user, num_item)
        elif args.pdf == 'Frechet':
            self.p = Frechet(num_user, num_item)
    '''
	[Input]	userAspRep:		bsz x num_aspects x h1
	[Input]	itemAspRep:		bsz x num_aspects x h1
	[Input]	userAspImpt:	bsz x num_aspects
	[Input]	itemAspImpt:	bsz x num_aspects
	[Input]	batch_uid:		bsz
	[Input]	batch_iid:		bsz
	'''
    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt, batch_uid, batch_iid,u_rating_ratio,i_rating_ratio,args):


        # User & Item Bias
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        lstAspRating = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)
        itemAspRep = torch.transpose(itemAspRep, 0, 1)

        for k in range(args.K):

            user = torch.unsqueeze(userAspRep[k], 1)
            item = torch.unsqueeze(itemAspRep[k], 2)
            aspRating = torch.matmul(user, item)
            aspRating = torch.squeeze(aspRating, 2)
            lstAspRating.append(aspRating)

        # List of (bsz x 1) -> (bsz x num_aspects)
        rating_pred = torch.cat(lstAspRating, dim = 1)
        #rating_pred = userAspImpt * itemAspImpt * rating_pred

        if args.pdf == 'blank':
            rating_pred = userAspImpt * itemAspImpt * rating_pred
            # Sum over all Aspects
            rating_pred = torch.sum(rating_pred, dim=1, keepdim=True).unsqueeze(1)
            # Include User Bias & Item Bias
            rating_pred = rating_pred + batch_userOffset + batch_itemOffset
            # Include Global Bias
            rating_pred = rating_pred + self.globalOffset

        else:
            rating_pred = torch.cat((userAspImpt, itemAspImpt, rating_pred), dim=-1)
            rating_pred = self.linear(rating_pred).unsqueeze(1)
            rating_pred = self.p(rating_pred, batch_uid, batch_iid, u_rating_ratio,i_rating_ratio)

        return rating_pred


class ANR(nn.Module):

    def __init__(self, word_weight_matrix,num_users, num_items,args,):

        super(ANR, self).__init__()

        # Word Embeddings (Input)
        self.wid_wEmbed = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.wid_wEmbed.weight.requires_grad = False
        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.shared_ANR_ARL = ANR_ARL(args)
        # Rating Prediction - Aspect Importance Estimation + Aspect-based Rating Prediction
        # Aspect-Based Co-Attention (Parallel Co-Attention, using the Affinity Matrix as a Feature) --- Aspect Importance Estimation
        self.ANR_AIE = ANR_AIE(args)
        # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
        self.ANR_RatingPred = ANR_RatingPred(num_users, num_items,args)


    def forward(self, batch_uid, batch_iid,user_review,item_review,u_rating_ratio,i_rating_ratio,args,item_list,user_list):

        # Embedding Layer
        batch_userDocEmbed = self.wid_wEmbed(user_review)
        batch_userDocEmbed = batch_userDocEmbed.reshape(len(batch_userDocEmbed),-1,300)
        batch_itemDocEmbed = self.wid_wEmbed(item_review)
        batch_itemDocEmbed = batch_itemDocEmbed.reshape(len(batch_itemDocEmbed),-1,300)

        userAspAttn, userAspDoc = self.shared_ANR_ARL(batch_userDocEmbed,args)
        itemAspAttn, itemAspDoc = self.shared_ANR_ARL(batch_itemDocEmbed,args)

        userCoAttn, itemCoAttn = self.ANR_AIE(userAspDoc, itemAspDoc)

        # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
        rating_pred = self.ANR_RatingPred(userAspDoc, itemAspDoc, userCoAttn, itemCoAttn, batch_uid, batch_iid,u_rating_ratio,i_rating_ratio,args)

        return rating_pred.squeeze()


class Gumbel(nn.Module):

    def __init__(self, num_user,num_item):

        super(Gumbel, self).__init__()

        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.mean_fun = nn.Linear(5, 5)
        self.var_fun = nn.Linear(5, 5)
        self.mlp1 = nn.Linear(5, 5)
        self.mlp2 = nn.Linear(5,5)


    def forward(self, feature,uid,iid,u_rating_ratio,i_rating_ratio):


        e = 2.71828182845
        rating = torch.FloatTensor([1.,2.,3.,4.,5.])
        rating = rating.cuda(0)

        umean = self.mean_fun(u_rating_ratio)
        uvar = self.var_fun(u_rating_ratio)
        uvar = (uvar)**2 + 0.2
        l1 = torch.pow(e,((rating-umean))/(uvar))
        l2 = torch.pow(e,-l1)
        out1 = 1/uvar * l1 * l2  #(32,1,5)

        imean = self.mean_fun(i_rating_ratio)
        ivar = self.var_fun(i_rating_ratio)
        ivar = (ivar)**2 + 0.2
        l1 = torch.pow(e, ((rating - imean)) / ivar)
        l2 = torch.pow(e, -l1)
        out2 = 1 / ivar * l1 * l2

        out = torch.cat((out1.unsqueeze(1), out2.unsqueeze(1), feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.mlp1(u_rating_ratio) # (32, 5)
        item_rating_ratio = self.mlp2(i_rating_ratio) # (32, 5)

        out = torch.sum(out*user_rating_ratio*item_rating_ratio, dim=-1)

        return out.squeeze()


class FM_Layer(nn.Module):

    def __init__(self,args):

        super(FM_Layer, self).__init__()
        self.linear = nn.Linear(2*args.n, 1)
        self.V = nn.Parameter(torch.zeros(2*args.n, 2*args.n))


    def forward(self, x):

        linear_part = self.linear(x)  #(32,1,160)
        V = torch.stack((self.V,) * x.size()[0]) #(32,160,160)

        # batch * 1 * input_dim
        interaction_part_1 = torch.bmm(x, V) #(32,1,160)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.bmm(torch.pow(x, 2), torch.pow(V, 2)) #(32,1,160)
        list_output = linear_part + torch.sum(0.5 * interaction_part_2 - interaction_part_1) #(32,1,1)
        rate = torch.stack(tuple(list_output), 0)

        return rate


class DeepConn(nn.Module):

    def __init__(self, word_weight_matrix,num_user,num_item,review_len,args):

        super(DeepConn, self).__init__()

        self.user_review_embeds = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.user_review_embeds.weight = nn.Parameter(word_weight_matrix, requires_grad=False)

        self.item_review_embeds = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.item_review_embeds.weight = nn.Parameter(word_weight_matrix, requires_grad=False)

        self.uconv2d = nn.Sequential(torch.nn.Conv2d(1, args.num_filters, \
                                                    kernel_size=(3,args.word_embedding), padding=(1,0) ), \
                                    nn.ReLU(),)

        self.iconv2d = nn.Sequential(torch.nn.Conv2d(1, args.num_filters, \
                                                    kernel_size=(3,args.word_embedding), padding=(1,0)), \
                                    nn.ReLU(),)

        self.get_ufea = nn.Linear(args.num_filters,args.n)
        self.get_ifea = nn.Linear(args.num_filters,args.n)

        self.fm = FM_Layer(args)

        self.linear = nn.Linear(2 *args.n,5)

        if args.pdf == 'Gumbel':
            self.p = Gumbel(num_user,num_item)

        elif args.pdf == 'GMM':
            self.p = GMM(num_user,num_item)

        elif args.pdf == 'Poisson':
            self.p = Poisson(num_user,num_item)

        elif args.pdf == 'Weibull':
            self.p = Weibull(num_user,num_item)

        elif args.pdf == 'Expon':
            self.p = Expon(num_user,num_item)

        elif args.pdf == 'Frechet':
            self.p = Frechet(num_user,num_item)

    def forward(self, batch_uid, batch_iid,user_review,item_review,u_rating_ratio,i_rating_ratio,args,item_list,user_list):

        user_review = self.user_review_embeds(user_review)# (batch,word_dim,review_len,review_doc_len)
        user_review = user_review.view(len(user_review),1,-1,args.word_embedding)
        user_review = self.uconv2d(user_review).squeeze(-1)
        user_review = F.max_pool1d(user_review,user_review.size()[2]).squeeze(-1)
        user_review = F.dropout(user_review,args.dropout)
        user_review = self.get_ufea(user_review)

        item_review = self.item_review_embeds(item_review)
        item_review = item_review.view(len(item_review), 1, -1, args.word_embedding)
        item_review = self.iconv2d(item_review).squeeze(-1)
        item_review = F.max_pool1d(item_review, item_review.size()[2]).squeeze(-1)
        user_review = F.dropout(user_review,args.dropout)
        item_review = self.get_ifea(item_review)

        feature = torch.cat((user_review,item_review),dim=-1).unsqueeze(1)

        feature = F.dropout(feature,args.dropout)

        if args.pdf == 'blank':
            rate = self.fm(feature)

        else:
            feature = self.linear(feature)
            rate = self.p(feature, batch_uid, batch_iid, u_rating_ratio, i_rating_ratio)

        return rate.squeeze()


class LocalAttention(nn.Module):
    def __init__(self, word_weight_matrix, args):
        super(LocalAttention, self).__init__()

        self.wid_wEmbed = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.wid_wEmbed.weight.requires_grad = False

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, args.word_embedding),padding = (2,0)),\
            nn.Sigmoid())

        self.cnn = nn.Sequential(
            nn.Conv2d(1, args.latt, kernel_size=(1, args.word_embedding)),
            nn.Tanh())


    def forward(self, x,args):

        review = self.wid_wEmbed(x) #(32,1,34102,100)
        review = review.view(len(review),1,-1,args.word_embedding)
        score = self.attention_layer(review) #(128,1,168,50)
        out = torch.mul(review,score) #(128,1,168,50)
        out = self.cnn(out) #(128,100,168)
        out = F.max_pool2d(out,(out.size()[2],1)) #(128,200,3,1)

        return out,review


class GlobalAttention(nn.Module):
    def __init__(self, review_len,args):
        super(GlobalAttention, self).__init__()

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(review_len, args.word_embedding)),
            nn.Sigmoid())

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(1,args.gatt, kernel_size=(2, args.word_embedding),padding=(1,0)),
            nn.Tanh())

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(1, args.gatt, kernel_size=(3, args.word_embedding),padding=(1,0)),
            nn.Tanh())

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(1, args.gatt, kernel_size=(4, args.word_embedding),padding=(2,0)),
            nn.Tanh())

    def forward(self, review,review_len,args):


        padding = torch.zeros(len(review),1,review_len-1,args.word_embedding)
        padding = padding.cuda()
        review1 = torch.cat((review,padding),dim= 2)

        score = self.attention_layer(review1) #(128,1,168,50) to (128,1,168,1)
        out = torch.mul(review, score)  #(128,1,168,50)
        out_1 = self.cnn_1(out)
        out_1 = F.max_pool2d(out_1,(out_1.size()[2],1)) #(128,100,168)
        out_2 = self.cnn_2(out)
        out_2 = F.max_pool2d(out_2, (out_2.size()[2],1))
        out_3 = self.cnn_3(out)
        out_3 = F.max_pool2d(out_3, (out_3.size()[2],1))

        return (out_1, out_2, out_3)  #(32,100,1,1)

class D_Att(nn.Module):

    def __init__(self,matrix,num_user,num_item,review_len,args):
        super(D_Att, self).__init__()

        self.review_len = review_len
        self.localAttentionLayer_user = LocalAttention(matrix, args)
        self.globalAttentionLayer_user = GlobalAttention(review_len,args)
        self.localAttentionLayer_item = LocalAttention(matrix, args)
        self.globalAttentionLayer_item = GlobalAttention(review_len,args)
        self.fcLayer = nn.Sequential(
            nn.Linear(500, 500),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(500, 50))

        self.linear = nn.Linear(50, 5)
        if args.pdf == "Gumbel":
            self.distribution = Gumbel(num_user,num_item)

        elif args.pdf == "Poisson":
            self.distribution = Poisson(num_user,num_item)

        elif args.pdf == 'Expon':
            self.p = Expon(num_user,num_item)

        elif args.pdf == 'GMM':
            self.distribution = GMM(num_user,num_item)

        elif args.pdf== 'Weibull':
            self.distribution = Weibull(num_user,num_item)

        elif args.pdf == 'Frechet':
            self.distribution = Frechet(num_user,num_item)


    def forward(self,batch_uid, batch_iid,user_review,item_review,u_rating_ratio,i_rating_ratio,args,item_list,user_list):

        #user
        local_user,user_review = self.localAttentionLayer_user(user_review,args) #(128,200,3,1), #(32,1,34102,100)
        global1_user, global2_user, global3_user = self.globalAttentionLayer_user(user_review,self.review_len,args)
        out_user = torch.cat((local_user, global1_user, global2_user, global3_user), 1)
        out_user = out_user.view(out_user.size(0), -1)
        out_user = self.fcLayer(out_user)

        # item
        local_item,item_review = self.localAttentionLayer_item(item_review,args) #(128,200,3,1), #(32,1,34102,100)
        global1_item, global2_item, global3_item = self.globalAttentionLayer_item(item_review,self.review_len,args) #(32,100,1,1)
        out_item = torch.cat((local_item, global1_item, global2_item, global3_item), 1) #(128,500,1,1)
        out_item = out_item.view(out_item.size(0), -1)
        out_item = self.fcLayer(out_item)

        if args.pdf == 'blank':

            out = torch.sum(torch.mul(out_user, out_item), 1)

        else:
            out = self.linear(out_user * out_item).unsqueeze(1)
            out = self.distribution(out, batch_uid, batch_iid, u_rating_ratio, i_rating_ratio)

        return out




class Weibull(nn.Module):

    def __init__(self, num_user,num_item):

        super(Weibull, self).__init__()

        self.alpha = nn.Linear(5,5)
        self.y = nn.Linear(5,5)

        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.layernorm = nn.BatchNorm1d(5,5)

        self.linear = nn.Linear(5,5)

        self.weight = nn.Parameter(torch.tensor([1.,2.,3.,4.,5.]),requires_grad=True)
        self.batchnorm = nn.BatchNorm1d(5)


    def forward(self,feature, batch_uid, batch_iid, u_rating_ratio, i_rating_ratio):

        e = 2.71828182845
        rating = torch.FloatTensor([1.,2.,3.,4.,5.])
        rating = rating.cuda()

        u_y = F.relu(self.alpha(u_rating_ratio))
        ualpha = F.relu(self.alpha(u_rating_ratio)) + 0.1
        out1 = u_y/ualpha * (rating/ualpha) ** (u_y-1) * torch.exp(-(rating/ualpha)**u_y)

        u_y = F.relu(self.alpha(i_rating_ratio))
        ualpha = F.relu(self.alpha(i_rating_ratio)) + 0.1
        out2 = u_y / ualpha * (rating / ualpha) ** (u_y - 1) * torch.exp(-(rating / ualpha) ** u_y)

        out = torch.cat((out1.unsqueeze(1),out2.unsqueeze(1),feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.linear(u_rating_ratio)
        item_rating_ratio = self.linear(i_rating_ratio)
        score = torch.sum(out*user_rating_ratio*item_rating_ratio, dim=-1)

        return score

class Frechet(nn.Module):

    def __init__(self, num_user,num_item,):

        super(Frechet, self).__init__()

        self.alpha = nn.Linear(5,5)
        self.shape_parameter = nn.Linear(5,5)
        self.y = nn.Linear(5,5)

        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.layernorm = nn.BatchNorm1d(5,5)

        self.linear = nn.Linear(5,5)

        self.weight = nn.Parameter(torch.tensor([1.,2.,3.,4.,5.]),requires_grad=True)
        self.batchnorm = nn.BatchNorm1d(5)


    def forward(self,feature, batch_uid, batch_iid, u_rating_ratio, i_rating_ratio):

        e = 2.71828182845
        rating = torch.FloatTensor([1.,2.,3.,4.,5.])
        rating = rating.cuda()

# User
        shape_u = F.relu(self.alpha(u_rating_ratio))
        beta_u = F.relu(self.alpha(u_rating_ratio)) + 0.1

        out1 = shape_u/beta_u*(rating/beta_u)**(shape_u-1) * torch.exp(-(rating/beta_u)**shape_u)
               #alpha_u/beta_u * (rating/beta_u) ** (-alpha_u-1) * torch.exp((rating/beta_u)**alpha_u)

# Item
        shape_i = F.relu(self.alpha(i_rating_ratio))
        beta_i = F.relu(self.alpha(i_rating_ratio)) + 0.1

        out2 = shape_i/beta_i*(rating/beta_i)**(shape_i-1) * torch.exp(-(rating/beta_i)**shape_i)
            #alpha_i / beta_i * (rating / beta_i) ** (-alpha_i - 1) * torch.exp(-(rating / beta_i) ** alpha_i)

        out = torch.cat((out1.unsqueeze(1), out2.unsqueeze(1),feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.linear(u_rating_ratio)
        item_rating_ratio = self.linear(i_rating_ratio)
        score = torch.sum(out*user_rating_ratio*item_rating_ratio, dim=-1)

        return score

class GMM(nn.Module):

    def __init__(self, num_user,num_item,):

        super(GMM, self).__init__()

        self.uid_var = nn.Embedding(num_user + 1, 5)
        self.uid_mean = nn.Embedding(num_user + 1, 5)

        self.iid_var  = nn.Embedding(num_item + 1, 5)
        self.iid_mean = nn.Embedding(num_item + 1, 5)

        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.layernorm = nn.BatchNorm1d(5,5)

        self.linear = nn.Linear(5,5)
        self.weight1 = nn.Parameter(torch.Tensor(5, 1),requires_grad=True)
        self.weight2 = nn.Parameter(torch.Tensor(5, 1), requires_grad=True)

        self.mean_fun = nn.Linear(5, 5)
        self.var_fun = nn.Linear(5, 5)


    def forward(self, feature,uid,iid,u_rating_ratio,i_rating_ratio):

        e = 2.71828182845
        pi = 3.14159265359
        #rating = torch.FloatTensor([[1.], [2.], [3.], [4.], [5.]])
        #rating = rating.expand(5,5)
        #rating = rating.cuda()
        rating = torch.FloatTensor([1., 2., 3., 4., 5.])
        rating = rating.cuda(0)

        umean = self.mean_fun(u_rating_ratio)
        uvar = self.var_fun(u_rating_ratio)
        uvar = uvar**2
        l1 = torch.pow(2*pi*uvar,1/2)
        l2 = -(rating-umean)**2/(2*uvar)
        out1 = 1/l1 * torch.pow(e,l2)

        imean = self.mean_fun(i_rating_ratio)
        ivar = self.var_fun(i_rating_ratio)
        ivar = ivar**2
        l1 = torch.pow(2 * pi * ivar , 1 / 2)
        l2 = -(rating - imean) ** 2 / (2 * ivar)
        out2 = 1 / l1 * torch.pow(e, l2)


        out = torch.cat((out1.unsqueeze(-1),out2.unsqueeze(-1),feature.permute(0, 2, 1)), dim=2)
        out = self.conv1d(out).squeeze(2)
        user_rating_ratio = self.linear(u_rating_ratio)
        item_rating_ratio = self.linear(i_rating_ratio)
        score = torch.sum(out*item_rating_ratio*user_rating_ratio, dim=-1)

        return score


class Expon(nn.Module):

    def __init__(self, num_user,num_item,):

        super(Expon, self).__init__()

        self.mean = nn.Linear(5, 1)
        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.layernorm = nn.BatchNorm1d(5, 5)
        self.linear = nn.Linear(5, 5)

    def forward(self, feature, uid, iid, u_rating_ratio, i_rating_ratio):

        rating = torch.FloatTensor([1., 2., 3., 4., 5.])
        rating = rating.cuda()

        ulambd = F.relu(self.mean(u_rating_ratio)) + 0.2
        out1 = ulambd * torch.exp(-ulambd * rating)

        ilambd = F.relu(self.mean(i_rating_ratio)) + 0.2
        out2 = ilambd * torch.exp(-ilambd * rating)

        out = torch.cat((out1.unsqueeze(1), out2.unsqueeze(1), feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.linear(u_rating_ratio)
        item_rating_ratio = self.linear(i_rating_ratio)
        score = torch.sum(out * user_rating_ratio * item_rating_ratio, dim=-1)

        return score.squeeze()


class Poisson(nn.Module):

    def __init__(self, num_user,num_item,):

        super(Poisson, self).__init__()

        self.mean = nn.Linear(5,1)

        self.conv1d = nn.Conv1d(5, 5, kernel_size=3)
        self.layernorm = nn.BatchNorm1d(5, 5)
        self.linear = nn.Linear(5,5)


    def forward(self, feature, uid, iid, u_rating_ratio,i_rating_ratio):


        e = 2.71828182845
        pi = 3.14159265359
        rating = torch.FloatTensor([1.,2.,3.,4.,5.])
        rating = rating.cuda()

        ulambd = F.relu(self.mean(u_rating_ratio)) + 0.2
        l0 = torch.pow(2*pi*rating,1/2) * torch.pow(rating/e,rating)
        l1 = torch.pow(ulambd,rating) * torch.pow(e,-ulambd)
        out1 = l1/l0

        ilambd = F.relu(self.mean(i_rating_ratio)) + 0.5
        l0 = torch.pow(2 * pi * rating, 1 / 2) * torch.pow(rating / e, rating)
        l1 = torch.pow(ilambd, rating) * torch.pow(e, -ilambd)
        out2 = l1 / l0

        out = torch.cat((out1.unsqueeze(1), out2.unsqueeze(1), feature), dim=1)
        out = self.conv1d(out.permute(0, 2, 1)).squeeze(2)
        user_rating_ratio = self.linear(u_rating_ratio)
        item_rating_ratio = self.linear(i_rating_ratio)
        score = torch.sum(out * user_rating_ratio * item_rating_ratio, dim=-1)

        return score.squeeze()


class NCF(nn.Module):

    def __init__(self,user_num,item_num,args):
        super(NCF, self).__init__()

        self.embed_user_GMF = nn.Embedding(user_num, args.factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, args.factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, args.factor_num * (2 ** (args.num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, args.factor_num * (2 ** (args.num_layers - 1)))

        MLP_modules = []
        for i in range(args.num_layers):
            input_size = args.factor_num * (2 ** (args.num_layers - i))
            MLP_modules.append(nn.Dropout(args.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(args.factor_num * 2, 1)

        self.linear = nn.Linear(2 * args.factor_num,5)
        self.gumbel = Gumbel(user_num,item_num)

        if args.pdf == 'GMM':
            self.p = GMM(user_num,item_num)
        elif args.pdf == 'Poisson':
            self.p = Poisson(user_num,item_num)
        elif args.pdf == 'Expon':
            self.p = Expon(user_num,item_num)
        elif args.pdf == 'Weibull':
            self.p = Weibull(user_num,item_num)
        elif args.pdf == 'Frechet':
            self.p = Frechet(user_num, item_num)
        elif args.pdf == 'Gumbel':
            self.p = Gumbel(user_num,item_num)

        self._init_weight_()

    def _init_weight_(self):

        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, batch_uid, batch_iid,user_review,item_review,u_rating_ratio,i_rating_ratio,args,item_list,user_list):

        embed_user_GMF = self.embed_user_GMF(batch_uid)
        embed_item_GMF = self.embed_item_GMF(batch_iid)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(batch_uid)
        embed_item_MLP = self.embed_item_MLP(batch_iid)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        if args.pdf == 'blank':
            prediction = self.predict_layer(concat).view(-1) #(32,1,2*num_factor)

        else:
            concat = self.linear(concat)
            prediction = self.p(concat,batch_uid,batch_iid,u_rating_ratio,i_rating_ratio)

        return prediction



class NRPA(nn.Module):

    def __init__(self, user_num, item_num, word_weight_matrix, review_len, args):

        super(NRPA, self).__init__()
        self.args = args
        self.review_embeds = nn.Embedding(word_weight_matrix.size(0), word_weight_matrix.size(1))
        self.review_embeds.weight = nn.Parameter(word_weight_matrix, requires_grad=False)

        self.ureview_conv2d = torch.nn.Conv2d(1, args.num_filters, \
                                            kernel_size=(3,args.word_embedding), padding=(1,0))

        self.ireview_conv2d = torch.nn.Conv2d(1, args.num_filters, \
                                            kernel_size=(3,args.word_embedding), padding=(1,0))

        self.uid_embeds = nn.Embedding(user_num+1, args.k2)
        self.iid_embeds = nn.Embedding(item_num + 1, args.k2)

        self.uidfm_embeds = nn.Embedding(user_num + 1, args.k2)
        self.iidfm_embeds = nn.Embedding(item_num + 1, args.k2)

        self.ufc_layer = nn.Linear(args.num_filters, args.k2)
        self.ifc_layer = nn.Linear(args.num_filters, args.k2)

        # user word/review level mlp
        self.u_w_linear = self.mlp_layer()
        self.u_r_linear = self.mlp_layer()
        # item word/review level mlp
        self.i_w_linear = self.mlp_layer()
        self.i_r_linear = self.mlp_layer()

        self.FM = FM_Layer()
        self.Wmul = nn.Linear(32, 1, bias=False)

        self.u_bias = nn.Embedding(user_num+1,1)
        torch.nn.init.constant_(self.u_bias.weight, 0.1)
        self.i_bias = nn.Embedding(item_num+1, 1)
        torch.nn.init.constant_(self.i_bias.weight, 0.1)
        self.globalbias = nn.Parameter(torch.Tensor(1),requires_grad=True)
        torch.nn.init.constant_(self.globalbias, 0.1)

        self.linear = nn.Linear(32,5)
        if args.pdf == 'Gumbel':
            self.p = Gumbel(user_num,item_num)
        elif args.pdf == 'Poisson':
            self.p = Poisson(user_num,item_num)
        elif args.pdf == 'Weibull':
            self.p = Weibull(user_num, item_num)
        elif args.pdf == 'Expon':
            self.p = Expon(user_num, item_num)
        elif args.pdf == 'Frechet':
            self.p = Frechet(user_num,item_num)
        elif args.pdf == 'GMM':
            self.p = GMM(user_num,item_num)


        self.dropout = nn.Dropout(args.dropout)

    def reset_weight(self):
        cnns = [self.word_item_cnn, self.word_user_cnn]
        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, a=-0.1, b=0.1)

        torch.nn.init.uniform_(self.Wmul.weight, -0.1, 0.1)
        nn.init.uniform_(self.uid_embeds.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.iid_embeds.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.uidfm_embeds.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.iidfm_embeds.weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.u_r_linear[0].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.u_r_linear[0].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.i_r_linear[-1].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.i_r_linear[-1].weight, a=-0.1, b=0.1)

        nn.init.uniform_(self.ufc_layer.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.ifc_layer.weight, a=-0.1, b=0.1)

        # linear = [self.ufc_layer, self.ifc_layer, self.u_w_linear[0], self.u_w_linear[-1], self.i_w_linear[0], self.i_w_linear[-1]]

    def mlp_layer(self):
        return nn.Sequential(
                    nn.Linear(self.args.k2, self.args.k2 * 2),
                    nn.LayerNorm(self.args.k2 * 2),
                    nn.ReLU(),
                    nn.Linear(self.args.k2 * 2, self.args.num_filters),
        )

    def forward(self, uid, iid, ureview, ireview, u_rating_ratio, i_rating_ratio, args,item_list,user_list):

        uid_feature = self.uid_embeds(uid)  #(batch,1,k2)
        iid_feature = self.iid_embeds(iid)
        uidfm = self.uidfm_embeds(uid)
        iidfm = self.iidfm_embeds(iid)
        # id_feature = torch.matmul(id_feature,self.Wu) #(batch,1,t)

        maxpool_size = ureview.size()[2]
        ureview = self.review_embeds(ureview)
        ureview_fea = ureview.view(-1, 1, maxpool_size, args.word_embedding)

        ureview_fea = self.ureview_conv2d(ureview_fea)
        ureview_fea = ureview_fea.view(len(ureview), -1, args.num_filters, maxpool_size, 1).squeeze()
        # ureview_fea = F.dropout(ureview_fea,args.dropout) #(128,review_len,num_filter)
        ureview_fea = F.max_pool2d(ureview_fea, (1, maxpool_size)).squeeze()  # (32,u_max,100)
        u_r_q = self.u_r_linear(uid_feature)#.permute(0, 2, 1)
        att_score = ureview_fea.bmm(u_r_q.permute(0,2,1))
        att_weight = F.softmax(att_score, 1)
        ufea = (ureview_fea * att_weight).sum(1, keepdim=True)
        ufea = self.ufc_layer(ufea) + uidfm
        # ufea = self.dropout(ufea)

        maxpool_size = ireview.size()[2]
        ireview = self.review_embeds(ireview)
        ireview_fea = ireview.view(-1, 1, maxpool_size, args.word_embedding)
        ireview_fea = self.ireview_conv2d(ireview_fea)  # (N,review_len,num_filters)
        # ireview_fea = F.dropout(ireview_fea) #(128,review_len,num_filter)
        ireview_fea = ireview_fea.view(len(ireview), -1, args.num_filters, maxpool_size, 1).squeeze()
        ireview_fea = F.max_pool2d(ireview_fea, (1, maxpool_size)).squeeze()  # (32,u_max,100)

        i_r_q = self.i_r_linear(iid_feature).permute(0, 2, 1)
        att_score = ireview_fea.bmm(i_r_q)
        att_weight = F.softmax(att_score, 1)
        ifea = (ireview_fea * att_weight).sum(1, keepdim=True)
        ifea = self.ifc_layer(ifea) + iidfm
        # ifea = self.dropout(ifea)

        if args.base == 'blank':
            score = self.Wmul((ufea*ifea))
            score = score + self.u_bias(uid) + self.i_bias(iid) + self.globalbias

        else:
            score = self.linear(ufea * ifea)
            score = self.p(score, uid, iid, u_rating_ratio, i_rating_ratio)

        return score.squeeze()



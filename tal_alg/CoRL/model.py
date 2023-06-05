import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MHSA_Intra(nn.Module):
    """
    compute intra-segment attention
    """
    def __init__(self, dim_in, heads, num_pos, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads
        self.num_pos = num_pos

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()
    
    def forward(self, input, intra_attn_mask):
        B, C, T = input.shape
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T) #(B, h, dim_head, T)
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B, h, T, dim_head)

        query *= self.scale
        sim = torch.matmul(query, key) #(B, h, T, T)
        intra_attn_mask = intra_attn_mask.view(B, 1, T, T)
        sim.masked_fill_(intra_attn_mask == 0, -np.inf)
        attn = F.softmax(sim, dim=-1) #(B, h, T, T)
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, value) #(B, h, T, dim_head)

        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T) #(B, C, T)
        output = input + self.bn(self.conv_out(output))
        return output


class CrossAttentionTransformer(nn.Module):
    def __init__(self,input_dim,dropout,heads,dim_feedforward=256):
        super(CrossAttentionTransformer,self).__init__()

        self.dim_in=input_dim
        self.dim_inner=self.dim_in
        self.dropout=dropout
        self.heads=heads
        self.dim_head = self.dim_inner // self.heads
        

        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0)
        self.w_out = nn.Linear(input_dim, input_dim)

        # self.self_atten = nn.MultiheadAttention(input_dim, num_heads=heads, dropout=0.1)

        # self.bn = nn.BatchNorm1d(num_features=self.dim_in, eps=1e-5, momentum=0.1)
        # self.bn.weight.data.zero_()
        # self.bn.bias.data.zero_()
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.proto=nn.Parameter(torch.zeros((20,5,768)),requires_grad=False)

    def forward(self,feature,prototypes=None,mask=None):
        if prototypes is None:
            prototypes=self.proto
        else:
            self.proto=nn.Parameter(prototypes,requires_grad=False)
        prototypes=prototypes.to(feature.device)
        B,D,T=feature.shape
        protos=prototypes.clone().view(1,D,-1).expand(B,-1,-1).contiguous()
        N=protos.shape[2]

        #cal Q,K,V
        query=feature.view(B, self.heads, self.dim_head, T).permute(0, 1, 3, 2).contiguous() #(B,h,T,dim_head)
        key=protos.view(B, self.heads, self.dim_head,-1) #(B, h, dim_head, N)
        value=protos.view(B, self.heads, self.dim_head, -1).permute(0, 1, 3, 2).contiguous()#(B, h, N, dim_head)
        #cal_attn
        query *= self.scale
        sim = torch.matmul(query, key) #(B, h, T, N)
        if mask is not None:
            mask=torch.matmul(mask.permute(1,0),torch.ones((1,N)).to(mask.device))
            mask = mask.view(B,1,T,N)
            sim.masked_fill_(mask == 0, -np.inf)
        attn = F.softmax(sim, dim=-1)#(B, h, T, N)
        # attn = torch.nan_to_num(attn, nan=0.0)

        output = torch.matmul(attn, value) #(B, h, T, dim_head)
        output = output.permute(0, 1, 3, 2).contiguous().view(B, D, T) #(B, C, T)
        # output = feature + self.bn(self.conv_out(output))
        feature = feature.permute(2, 0, 1)
        output = output.permute(2, 0, 1)

        
        # feature=feature.permute(2, 0, 1)
        # q=feature
        # k=protos.view(B,D,-1).permute(2,0,1)
        # v=protos.view(B,D,-1).permute(2,0,1)
        # output,_=self.self_atten(q, k, v)

        #FFN
        feature = feature + self.dropout1(output)
        feature = self.norm1(feature)
        feature2 = self.linear2(self.dropout(F.relu(self.linear1(feature))))
        feature = feature + self.dropout2(feature2)
        feature = self.norm2(feature)

        feature = feature.permute(1,2,0)
        return feature,attn

        

class SelfAttentionTransformer(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8,dim_feedforward=128, pos_embed=False):
        super(SelfAttentionTransformer, self).__init__()
        self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)

        dim_feedforward = dim_feedforward
        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.w_out = nn.Linear(input_dim, input_dim)

    def forward(self, features, attn_mask=None, pos_embed=None, pre_LN=False):
        src = features.permute(2, 0, 1)
        if pos_embed is not None:
            q = k = src + pos_embed
        else:
            q = k = src
            q=self.conv_query(features).permute(2,0,1)
            k=self.conv_key(features).permute(2,0,1)

        if pre_LN:  
            src2,attn = self.self_atten(q, k, self.norm1(src))
            src = src + self.dropout1(src2)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(src2)
        else:
            src2,attn = self.self_atten(q, k, src, attn_mask=attn_mask)
            
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            # src = src+self.dropout(self.w_out(src2))
            # src = self.norm2(src)

        src = src.permute(1, 2, 0)
        return src,attn
    


class PGM(nn.Module):
    def __init__(self,args):
        super(PGM,self).__init__()
        self.feature_dim=args.feature_dim
        self.drop_thresh=0.7
        self.point_detector=nn.Sequential(
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(self.drop_thresh),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )
    def forward(self,x):
        input_features=x.permute(0,2,1)#(B,F,T)
        p_heatmap=self.point_detector(input_features).permute(0,2,1)#(B,T,1)
        return p_heatmap


class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.dataset=args.dataset
        self.feature_dim=args.feature_dim
        self.action_cls_num=args.action_cls_num
        self.drop_thresh = args.dropout
        self.r_act=args.r_act

        self.BaS=args.BaS
        self.part_topk=args.part_topk
        self.transformer_args=args.transformer_args

        # self.mask_attn = MHSA_Intra(dim_in=self.feature_dim, num_pos=args.num_segments, heads=self.transformer_args['num_heads'])

        self.self_attn =nn.ModuleList([SelfAttentionTransformer(
            self.feature_dim, self.transformer_args['drop_out'],
            num_heads=self.transformer_args['num_heads'],
            dim_feedforward=self.transformer_args['dim_feedforward']
            ) for i in range(self.transformer_args['layer_num'])]) 
    
        num_cross_attn=1
        self.cross_attn=nn.ModuleList([CrossAttentionTransformer(input_dim=self.feature_dim, dropout=0.3, heads=8) for i in range(num_cross_attn)])

        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS14":
            self.feature_embedding=nn.Sequential(
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        elif self.dataset == "ActivityNet13":
            self.feature_embedding = nn.Sequential(
                nn.Dropout(self.drop_thresh),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                )
        else:
            self.feature_embedding=nn.Sequential(
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        
        self.classifier=nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.action_cls_num+1, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        self.softmaxd1=nn.Softmax(dim=1)

    def forward(self,input_features,prototypes,vid_labels=None,temporal_mask=None): #One-branch
        #input_feature:(B,T,F)
        B,T,F=input_features.shape
        num_segment=input_features.shape[1]
        k_act=max(1,num_segment//self.r_act)
        input_features=input_features.permute(0,2,1)#(B,F,T)
        
        if temporal_mask is not None:
            temporal_mask=temporal_mask.to(input_features.device)
            attn_mask=torch.matmul(temporal_mask.permute(1,0),temporal_mask)

        if hasattr(self, 'mask_attn'):
            input_features=self.mask_attn(input_features,attn_mask)


        if hasattr(self, 'self_attn'):
            for layer in self.self_attn:
                input_features,self_attn = layer(input_features)
                # input_features = layer(input_features,attn_mask)

        if hasattr(self,'cross_attn'):
            for layer in self.cross_attn:
                input_features,cross_attn = layer(input_features,prototypes)


        input_features=self.feature_embedding(input_features)#(B,F,T)
        embeded_feature_base = input_features.permute(0,2,1) #(B,T,F)
        cas_base=self.classifier(self.dropout(input_features)) #(B,C+1,T)
        cas_base=cas_base.permute(0,2,1) #(B,T,C+1)
        cas_sigmoid_base=self.sigmoid(cas_base)
        
        value_base,_=cas_sigmoid_base.sort(descending=True,dim=1)
        topk_scores_base=value_base[:,:k_act,:-1]
        if vid_labels is None:
            vid_score_base = torch.mean(topk_scores_base, dim=1)
        else:
            vid_score_base = (torch.mean(topk_scores_base, dim=1) * vid_labels) + (torch.mean(cas_sigmoid_base[:,:,:-1], dim=1) * (1 - vid_labels))
        cas_sigmoid_fuse_base=cas_sigmoid_base[:,:,:-1] * (1 - cas_sigmoid_base[:,:,-1].unsqueeze(2))
        cas_sigmoid_fuse_base = torch.cat((cas_sigmoid_fuse_base, cas_sigmoid_base[:,:,-1].unsqueeze(2)), dim=2)


        return vid_score_base,embeded_feature_base,cas_sigmoid_fuse_base,\
                vid_score_base,embeded_feature_base,cas_sigmoid_fuse_base



def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    disc_occupy=total*4/1024/1024 #MB
    print("Total para_nums:{},disc_occupied:{}MB".format(total,disc_occupy))

if __name__=="__main__":
    import options
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    args=options.parse_args()
    model=Model(args)
    print_model_parm_nums(model)
    print(parameter_count_table(model))




class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        # 定义可训练参数，即论文中的W和a
        self.fc_q = nn.Linear(in_features, out_features)
        self.fc_k = nn.Linear(in_features, out_features)
        self.fc_v = nn.Linear(in_features, out_features)

        self.self_attn_layer_norm = nn.LayerNorm(out_features)
        self.scale = torch.sqrt(torch.FloatTensor([self.out_features])).to('cuda:0')

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, w_adj, Dif,Eyc):

        redis = inp


        inp = Eyc.squeeze() @ inp

        Q = self.fc_q(inp)
        K = self.fc_k(inp)
        V = self.fc_v(inp)

        K = K.permute(0, 2, 1)

        e = torch.matmul(Q, K) / (self.scale)


       

        e = w_adj * e + (1 - w_adj) * Dif
        e = F.softmax(e, dim=-1)


        x = (e + e @ e) @ V

        x = F.relu(x)

        x = x + redis
        x = self.self_attn_layer_norm(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class dif_lin(nn.Module):
    def __init__(self):
        super(dif_lin, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=(1, 1), bias=True)

    def forward(self, adjs,Eyc):
        deta = 0


        #print(adjs.shape)
        #print(Eyc.shape)
        aes = Eyc @ adjs
        for i in range(adjs.size(1) - 1):
            deta = deta + self.conv((aes[:, i + 1, :, :]   - aes[:, i, :, :]).unsqueeze(1))

        return deta.squeeze()


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs):
        batch = node_embs.size(0)

        scores = node_embs.matmul(self.scorer) / self.scorer.norm()

        vals, topk_indices = scores.view(-1,node_num).topk(self.k,dim=-1)


        zore = torch.zeros(batch ,node_num).to("cuda:0")

        zore.scatter_(1,topk_indices,1)

        Eyc = torch.diag_embed(zore)
        Eyc = Eyc.view(batch,-1,node_num,node_num)
        #print(Eyc.shape)
        # we need to transpose the output
        return Eyc


class gwnet(nn.Module):
    def __init__(self, dropout=0.3, adj_input=4, in_dim=12, out_dim=32, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.residual_channels = residual_channels
        self.dilation = []


        self.Tk = TopK(residual_channels,20)
        # self.Gcn = GraphConvolution(6,residual_channels,residual_channels)
        # def __init__(self, in_features, out_features, dropout, alpha, concat=True):

        # self.GAT = GraphAttentionLayer(residual_channels, residual_channels, 8,0.5, 0.3)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.GAT = nn.ModuleList()
        self.gate_attention = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.dif_set = nn.ModuleList()

        self.Time_weights = nn.Parameter(torch.FloatTensor(Time_length, node_num, node_num))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1), bias=True)

        self.GNN_Concat = nn.Conv2d(in_channels=2 * residual_channels,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.Graph_Start = nn.Conv2d(in_channels=adj_input, out_channels=residual_channels, kernel_size=(1, 2),
                                     bias=True)
        # self.Graph_T = nn.Conv2d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=(1, 2))
        receptive_field = 1
        receptive_field_p = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                receptive_field_p = receptive_field_p + new_dilation

                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=receptive_field_p,
                                                 out_channels=1,
                                                 kernel_size=(1, 1), bias=True))

                self.GAT.append(GraphAttentionLayer(residual_channels, residual_channels, 8, 0.5, 0.3))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.dif_set.append(dif_lin())

                # 1x1 convolution for skip connection
                self.gate_attention.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.dilation.append(new_dilation)
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Time_weights)

    def forward(self, input):
        in_len = input.size(1)
        batch_size = input.size(0)
        if input.size(1) < self.receptive_field:
            x = nn.functional.pad(input, (0, 0, 0, 0, self.receptive_field - input.size(1), 0))
        else:
            x = input

        x = x.permute(0, 3, 2, 1)

        x = self.start_conv(x)
        skip = 0

        # WaveNet layers

        dlation_sum = 1
        for i in range(self.blocks * self.layers):

            residual = x

            # dilated convolution
            x = self.filter_convs[i](residual)
            x = F.gelu(x)

            x = x.permute(0, 3, 2, 1)
            in_len = in_len - self.dilation[i]
            dlation_sum = dlation_sum + self.dilation[i]
            gat_output = []

            for j in range(in_len):

                Eyc = self.Tk(x[:,j,:,:])
                #Eyc = 1

                w_adj = self.gate_convs[i](input[:, j:j + dlation_sum, :, :])
                w_adj = w_adj.squeeze()
                w_adj = F.sigmoid(w_adj)

                Dif = self.dif_set[i](input[:, j:j + dlation_sum, :, :],Eyc)

                gat_output.append(self.GAT[i](x[:, j, :, :], w_adj, Dif,Eyc))

            gat_output = torch.stack(gat_output)
            # print(gat_output.shape)
            x = gat_output.permute(1, 0, 2, 3)
            x = x.permute(0, 3, 2, 1)

            attention = self.gate_attention[i](x)
            attention = attention.permute(0, 3, 2, 1)
            redis_att = attention
            attention = (attention.sum(-2)/node_num).unsqueeze(-2)
            attention = torch.softmax(attention,dim=-1)
            #print(attention.shape)
            #print(x.shape)
            attention = redis_att + attention
            attention = F.sigmoid(attention)

            x = x.permute(0, 3, 2, 1)
            x = x * attention
            x = F.gelu(x)

            x = x.permute(0, 3, 2, 1)

            #print(x.shape)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        emb = skip.permute(0, 3, 2, 1)
        emb = emb.squeeze()

        x = F.relu(self.end_conv_1(skip))
        x = self.end_conv_2(x)
        x = F.gelu(x)
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        return x, emb

import torch


class EGES(torch.nn.Module):
    def __init__(self, embedding_dim, num_nodes, num_brands, num_shops, num_cates):
        super(EGES, self).__init__()
        self.embedding_dim = embedding_dim
        # embeddings for nodes itself(input embedding)
        base_in_embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        # output embedding
        self.base_out_embedding = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)

        # SI brand
        brand_embedding = torch.nn.Embedding(num_brands, embedding_dim)
        # SI shop
        shop_embedding = torch.nn.Embedding(num_shops, embedding_dim)
        # SI cate
        cate_embedding = torch.nn.Embedding(num_cates, embedding_dim)
        # types of side information(including itself)
        self.si_nums = 4
        self.embeddings = torch.nn.ModuleList([base_in_embedding, brand_embedding, shop_embedding, cate_embedding])
        # weights Matrix A shape: |V| * si_nums
        # every item has different si weight contribution
        self.si_weights = torch.nn.Embedding(num_nodes, self.si_nums)

    def forward(self, node1, node2):
        # node: [sku_id, brand_id, shop_id, cate_id]
        # node1_embedding: batch_size * embedding_dim (input vector space)
        node1_embedding = self.get_si_weighted_embedding(node1)
        # node2_embedding: batch_size * embedding_dim (output vector space)
        node2_embedding = self.base_out_embedding(node2[:, 0])
        return node1_embedding, node2_embedding

    def get_si_weighted_embedding(self, x):
        """
            @x: batch_size * SI's num (sku_id, brand_id, shop_id, cate_id)
            hidden: batch_size * embedding_dim
        """
        if x.is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        batch_size = x.shape[0]
        # x[:, 0] is a list of sku_id of the batch (batch_size,)
        batch_si_weights = torch.exp(self.si_weights(x[:, 0]))
        si_weighted_embedding = torch.zeros((batch_size, self.embedding_dim)).to(device)
        si_weight = torch.zeros((batch_size, 1)).to(device)
        for i in range(self.si_nums):
            # (batch_size, ) -> (batch_size, 1)
            batch_i_th_si_weight = batch_si_weights[:, i].view((batch_size, 1))
            # self:(batch_size, dim) + ... + cate:(batch_size, dim)
            si_weighted_embedding += batch_i_th_si_weight * self.embeddings[i](x[:, i])
            # self:(batch_size, 1) + ... + cate:(batch_size, 1)
            si_weight += batch_i_th_si_weight
        # (batch_size, dim) / (batch_size, 1) (element-wise)
        hidden = si_weighted_embedding / si_weight
        return hidden

    def loss(self, node1_embedding, node2_embedding, labels):
        # cross-entropy loss
        dots = torch.sigmoid(torch.sum(node1_embedding * node2_embedding, axis=1))
        # maybe log 0 and result in nan,so clamp
        # https://stackoverflow.com/questions/65310095/getting-nan-as-loss-value
        dots = torch.clamp(dots, min=1e-7, max=1 - 1e-7)
        all_loss = -labels * torch.log(dots) - (1 - labels) * torch.log(1 - dots)
        batch_mean_loss = torch.mean(all_loss)
        return batch_mean_loss

    def get_cold_item_embedding(self, x):
        """
            @x: batch_size * SI's num (sku_id, brand_id, shop_id, cate_id)
            paper3.3.2: we represent a cold start item with the average embeddings of its side information
        """
        if x.is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        si_weighted_embedding = torch.zeros(x.shape[0], self.embedding_dim).to(device)
        si_weighted_embedding += self.embeddings[1](x[:, 1])
        si_weighted_embedding += self.embeddings[2](x[:, 2])
        si_weighted_embedding += self.embeddings[3](x[:, 3])
        return si_weighted_embedding / 3

import copy

import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from graph import RandWalk
from model import EGES


def train_and_save(train_g, test_g, encode_sku_info,
                   num_skus, num_brands, num_shops, num_cates,
                   num_walks, walk_length, window_size, num_negative,
                   batch_size, embedding_dim, lr, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    walker = RandWalk(train_g, walk_length, num_walks, window_size, num_negative)
    # for each node in the graph, we sample pos and neg
    # pairs for it, and feed these sampled pairs into the model.
    # (nodes in the graph are of course batched before sampling)
    dataloader = DataLoader(
        torch.arange(train_g.num_nodes()),
        # this is the batch_size of input nodes
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: walker.transfer(x, encode_sku_info)
    )
    model = EGES(embedding_dim, num_skus, num_brands, num_shops, num_cates).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_list = []
    for epoch in range(epochs):
        epoch_total_loss = 0
        for step, (srcs, dsts, labels) in enumerate(dataloader):
            # the batch size of output pairs is unfixed 因为随机游走生成的序列长度是不确定的
            srcs_embeds, dsts_embeds = model(srcs.to(device), dsts.to(device))
            loss = model.loss(srcs_embeds, dsts_embeds, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()

        print('Epoch{}: Loss {:05f}'.format(epoch, epoch_total_loss))
        # 深拷贝 https://stackoverflow.com/questions/56526698/best-state-changes-with-the-model-during-training-in-pytorch
        model_list.append((epoch, copy.deepcopy(model.state_dict()), eval_link_prediction(model, test_g, encode_sku_info, device)))

    model_list.sort(key=lambda x: x[2], reverse=True)
    print(model_list[0])
    torch.save(model_list[0][1], str(embedding_dim) + '_' + f"{model_list[0][2]:.4f}" + "_eges_model.pt")


def eval_link_prediction(model, test_graph, encode_sku_info, device):
    # model.to(device)
    with torch.no_grad():
        preds, labels = [], []
        for src, dst, label in test_graph:
            src = torch.tensor(encode_sku_info[src.item()]).view(1, 4).to(device)
            dst = torch.tensor(encode_sku_info[dst.item()]).view(1, 4).to(device)
            # (1, dim)
            src = model.get_si_weighted_embedding(src)
            dst = model.get_si_weighted_embedding(dst)
            # (1, dim) -> (1, dim) -> (1, )
            logit = torch.sigmoid(torch.sum(src * dst))
            preds.append(logit.item())
            labels.append(label)

        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)

        print("Evaluate link prediction AUC: {:.4f}".format(metrics.auc(fpr, tpr)))
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    import utils, graph

    item_info_data = "data/jdata_product.csv"
    action_data = "data/action_head.csv"
    session_interval_sec = 7200
    min_sku_freq = 15
    batch_size = 128
    epochs = 10
    lr = 0.1

    num_walks = 5
    walk_length = 10
    window_size = 2
    num_negative = 5
    embedding_dim = 16

    all_sku_info = utils.get_all_sku_si_set(item_info_data)

    # g 为 DGL的图结构
    g, sku_info_encoder, sku_info_decoder, encode_sku_info = graph.construct_item_graph(action_data, session_interval_sec, all_sku_info.keys(), min_sku_freq, all_sku_info)

    train_g, test_edges = graph.split_train_test_graph(g, num_negative)

    num_skus = train_g.num_nodes()
    num_brands = len(sku_info_encoder["brand"])
    num_shops = len(sku_info_encoder["shop"])
    num_cates = len(sku_info_encoder["cate"])

    print(f"Num skus: {num_skus}, num brands: {num_brands}, num shops: {num_shops}, num cates: {num_cates}")

    train_and_save(train_g, test_edges, encode_sku_info, num_skus, num_brands, num_shops, num_cates, num_walks, walk_length, window_size, num_negative, batch_size, embedding_dim, lr, epochs)

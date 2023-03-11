def get_all_sku_si_set(item_info_path):
    all_sku_info = {}
    with open(item_info_path, "r", encoding='utf-8-sig') as f:
        f.readline()
        for line in f.readlines():
            alist = line.split(",")
            all_sku_info[alist[0]] = (alist[1], alist[2], alist[3])
    return all_sku_info


# 对列表内元素进行映射编码
def encode(alist):
    aset = set(alist)
    encoder = {}
    decoder = {}
    for idx, sku in enumerate(aset, 0):
        encoder[sku] = idx
        decoder[idx] = sku
    return encoder, decoder

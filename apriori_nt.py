from itertools import combinations

def find_frequent_itemsets(transactions, min_support):
    # Tính tần suất xuất hiện của từng mặt hàng đơn lẻ
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1

    # Lọc ra các mặt hàng có tần suất hỗ trợ lớn hơn hoặc bằng min_support
    frequent_items = {item for item, count in item_counts.items() if count >= min_support}

    # Bắt đầu từ các tập itemset kích thước 2 trở lên và kiểm tra tần suất
    k = 2
    frequent_itemsets = {frozenset([item]) for item in frequent_items}
    while True:
        candidate_itemsets = set(combinations(frequent_items, k))
        frequent_itemsets_this_round = set()
        for itemset in candidate_itemsets:
            support_count = 0
            for transaction in transactions:
                if set(itemset).issubset(transaction):
                    support_count += 1
            if support_count >= min_support:
                frequent_itemsets_this_round.add(frozenset(itemset))
        if not frequent_itemsets_this_round:
            break
        frequent_itemsets.update(frequent_itemsets_this_round)
        k += 1

    return frequent_itemsets


transactions = [
    {'book', 'clock', 'curtain'},
    {'book', 'clock', 'tv'},
    {'book', 'painting', 'vase'},
    {'clock', 'curtain', 'painting'},
    {'clock', 'tv'},
    {'curtain', 'vase'},
    {'painting', 'vase'},
    {'painting', 'tv'},
]

min_support = 2
frequent_itemsets = find_frequent_itemsets(transactions, min_support)
for itemset in frequent_itemsets:
    print(itemset)

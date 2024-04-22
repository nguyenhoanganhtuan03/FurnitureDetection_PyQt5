from mlxtend.frequent_patterns import apriori
import pandas as pd

def convert_transactions(transactions):
    # Chuyển đổi danh sách giao dịch thành dạng DataFrame
    converted_transactions = []
    for transaction in transactions:
        transaction_dict = {}
        for item in transaction:
            transaction_dict[item] = True
        converted_transactions.append(transaction_dict)
    df = pd.DataFrame(converted_transactions).fillna(False)
    return df


def find_frequent_itemsets(transactions, min_support):
    # Tìm tập phổ biến sử dụng thuật toán Apriori
    df = convert_transactions(transactions)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def extended_transactions(transactions, additional_items):
    # Mở rộng các giao dịch ban đầu bằng cách thêm các mục mới từ additional_items
    extended_transactions = []
    for transaction in transactions:
        for new_items in additional_items.values():  # Lặp qua các giá trị của từ điển
            extended_transactions.append(transaction | set(new_items))
    return extended_transactions

def suggest_items(input_items, frequent_itemsets):
    suggestions = []
    # Lặp qua các tập mục phổ biến
    for _, itemset in frequent_itemsets.iterrows():
        # Kiểm tra nếu tập mục này chứa ít nhất một mặt hàng từ tập đầu vào
        if input_items.intersection(set(itemset['itemsets'])):
            # Lấy tất cả các mặt hàng không thuộc tập đầu vào
            suggested_items = set(itemset['itemsets']) - input_items
            suggestions.append(suggested_items)
    return suggestions


transactions = [{'book', 'clock', 'curtain'}, {'book', 'clock', 'tv'},
                {'book', 'painting', 'vase'}, {'clock', 'curtain', 'painting'},
                {'clock', 'tv'}, {'curtain', 'vase'},
                {'painting', 'vase'}, {'painting', 'tv'}]

additional_items = {'lamp':'Đèn', 'rug': 'Tấm thảm', 'mirror': 'Gương', 'candle': 'Nến', 'chair': 'Ghế', 'table': 'Bàn', 'sofa': 'Ghế Sofa'}
additional = [{'lamp', 'rug'}, {'mirror', 'candle'}, {'chair', 'table'}, {'sofa', 'rug'}]

# Tính toán tập phổ biến sử dụng hàm find_frequent_itemsets
frequent_itemsets = find_frequent_itemsets(extended_transactions(transactions, additional_items), min_support=0.2)

# # In các tập phổ biến
# print("Các tập mục phổ biến với các giao dịch mở rộng:")
# print(frequent_itemsets)
#
# # Gợi ý dựa trên các mặt hàng đã cho
# input_items = {'book', 'clock'}
# suggestions = suggest_items(input_items, frequent_itemsets)
# print("Các gợi ý dựa trên các mặt hàng đã cho:")
# for suggestion in suggestions:
#     print(suggestion)

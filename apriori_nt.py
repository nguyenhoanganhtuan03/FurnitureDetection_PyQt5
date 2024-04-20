from mlxtend.frequent_patterns import apriori
import pandas as pd

def convert_transactions(transactions):
    converted_transactions = []
    for transaction in transactions:
        transaction_dict = {}
        for item in transaction:
            transaction_dict[item] = 1
        converted_transactions.append(transaction_dict)
    df = pd.DataFrame(converted_transactions).fillna(0)
    return df

def find_frequent_itemsets(transactions, min_support):
    df = convert_transactions(transactions)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# def main():
#     transactions = [{'book', 'clock', 'curtain'}, {'book', 'clock', 'tv'},
#                     {'book', 'painting', 'vase'}, {'clock', 'curtain', 'painting'},
#                     {'clock', 'tv'}, {'curtain', 'vase'},
#                     {'painting', 'vase'}, {'painting', 'tv'}]
#
#     additional_items = [{'lamp', 'rug'}, {'mirror', 'candle'}, {'chair', 'table'}, {'sofa', 'rug'}]
#
#     # Mở rộng giao dịch
#     extended_transactions = []
#     for transaction in transactions:
#         for new_items in additional_items:
#             extended_transactions.append(transaction | new_items)
#
#     # Tính toán tập phổ biến sử dụng hàm find_frequent_itemsets
#     frequent_itemsets = find_frequent_itemsets(extended_transactions, min_support=0.2)
#
#     # In các tập phổ biến
#     print("Frequent Itemsets with extended transactions:")
#     print(frequent_itemsets)
#
# if __name__ == "__main__":
#     main()

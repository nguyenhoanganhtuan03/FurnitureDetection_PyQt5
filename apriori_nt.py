from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

def convert_transactions(transactions):
    # Tạo một danh sách trống để lưu trữ giao dịch đã chuyển đổi
    converted_transactions = []

    # Lặp qua mỗi giao dịch
    for transaction in transactions:
        # Tạo một dict để lưu trữ sự xuất hiện của mỗi mặt hàng trong giao dịch
        transaction_dict = {}

        # Lặp qua mỗi mặt hàng và đánh dấu sự xuất hiện của mỗi mặt hàng trong giao dịch
        for item in transaction:
            transaction_dict[item] = 1

        # Thêm giao dịch đã chuyển đổi vào danh sách giao dịch đã chuyển đổi
        converted_transactions.append(transaction_dict)

    # Tạo DataFrame từ danh sách giao dịch đã chuyển đổi
    df = pd.DataFrame(converted_transactions).fillna(0)

    return df

def main():
    transactions = [{'book', 'clock', 'curtain'}, {'book', 'clock', 'tv'},
                    {'book', 'painting', 'vase'}, {'clock', 'curtain', 'painting'},
                    {'clock', 'tv'}, {'curtain', 'vase'},
                    {'painting', 'vase'}, {'painting', 'tv'}]

    additional_items = [{'lamp', 'rug'}, {'mirror', 'candle'}, {'chair', 'table'}, {'sofa', 'rug'}]

    # Mở rộng giao dịch
    extended_transactions = []
    for transaction in transactions:
        for new_items in additional_items:
            extended_transactions.append(transaction | new_items)

    # Chuyển đổi giao dịch thành định dạng DataFrame cho `mlxtend`
    df = convert_transactions(extended_transactions)

    # Tính toán tập phổ biến sử dụng thuật toán Apriori
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

    # In các tập phổ biến
    print("Frequent Itemsets with extended transactions:")
    print(frequent_itemsets)

if __name__ == "__main__":
    main()

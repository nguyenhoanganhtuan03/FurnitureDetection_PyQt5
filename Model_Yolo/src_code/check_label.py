import os

def check_label(label_file):
    # Đọc nội dung của file nhãn
    with open(label_file, 'r') as f:
        content = f.read().strip().split()

    # Kiểm tra xem nội dung có đúng hay không
    try:
        # Lấy 4 số cuối cùng từ nội dung file nhãn và chuyển đổi sang kiểu float
        last_four_digits = [float(x) for x in content[-4:]]
        # Kiểm tra xem tất cả các số có nhỏ hơn 1 không
        for digit in last_four_digits:
            if digit > 1:
                return False
        return True
    except ValueError:
        return False

def remove_label_and_image(label_file, image_files, folder_path):
    # Xóa cả nhãn và hình tương ứng
    os.remove(label_file)
    label_filename = os.path.splitext(os.path.basename(label_file))[0]  # Lấy tên tệp nhãn mà không có phần mở rộng và đường dẫn
    for image_file in image_files:
        image_filename = os.path.splitext(os.path.basename(image_file))[0]  # Lấy tên tệp hình mà không có phần mở rộng và đường dẫn
        if image_filename == label_filename:
            os.remove(image_file)


def main():
    folder_path = "dataset/val/5_tv"
    label_extension = ".txt"
    image_extension = [".jpg", ".png", ".jpeg"]

    label_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(label_extension)]
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if any(file.endswith(ext) for ext in image_extension)]

    for label_file in label_files:
        if not check_label(label_file):
            remove_label_and_image(label_file, image_files, folder_path)
            print(f"Đã xóa nhãn và hình tương ứng của {label_file}")

if __name__ == "__main__":
    main()

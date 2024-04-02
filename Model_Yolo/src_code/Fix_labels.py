import os

kt_tt = '0'

def sua_nhan_trong_thu_muc(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu nó không tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Duyệt qua tất cả các tệp tin trong thư mục đầu vào
    for file_name in os.listdir(input_folder):
        try:
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            with open(input_file_path, 'r') as input_file:
                lines = input_file.readlines()

            # Sửa nhãn
            kt = lines[0][0]
            lines = [line.replace(kt, kt_tt,1) for line in lines]

            with open(output_file_path, 'w') as output_file:
                output_file.writelines(lines)
            print(f"Đã sửa nhãn trong {file_name} và lưu vào {output_file_path}")
        except Exception as e:
            print(f"\tCó lỗi xảy ra khi xử lý tập tin {file_name}: {e}")
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(lines)
            continue

# Sử dụng hàm:
input_folder = "labels"  # Thay đổi thành đường dẫn thực tế của thư mục đầu vào
output_folder = "fixed"  # Thay đổi thành đường dẫn thực tế của thư mục đầu ra
sua_nhan_trong_thu_muc(input_folder, output_folder)


import os


def count_matching_files(folder_path, label_extension, image_extensions):
    label_files = [file for file in os.listdir(folder_path) if file.endswith(label_extension)]
    image_files = [file for file in os.listdir(folder_path) if any(file.endswith(ext) for ext in image_extensions)]

    unmatched_images = []

    if len(label_files) != len(image_files):
        # Tìm các file hình không có nhãn
        unmatched_images = [image_file for image_file in image_files if
                            image_file.replace(label_extension, "") not in label_files]

    matching_count = 0

    for label_file in label_files:
        image_file = label_file.replace(label_extension, "")
        if image_file in image_files:
            matching_count += 1

    return matching_count, len(image_files), len(label_files), unmatched_images


def main():
    folder_path = "dataset/val"
    label_extension = ".txt"
    image_extensions = [".jpg", ".png", ".jpeg"]

    matching_count, total_images, total_labels, unmatched_images = count_matching_files(folder_path, label_extension, image_extensions)

    if matching_count is not None and total_labels is not None:
        print(f"Tổng số nhãn và hình: {total_images+total_labels}")
        print(f"Tổng số file nhãn: {total_labels}")
        print(f"Số lượng file hình có nhãn tương ứng: {total_images}")

        if total_images == total_labels:
            print("Mỗi file nhãn đều có file hình tương ứng.")

        if unmatched_images:
            print("Các file hình không có nhãn tương ứng:")
            for image in unmatched_images:
                print(image)


if __name__ == "__main__":
    main()


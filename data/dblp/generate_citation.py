def process_dataset(input_file, output_file, threshold=85636):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 移除第一列
            parts = line.strip().split()[1:]
            # 過濾掉大於 threshold 的數字
            filtered = [x for x in parts if int(x) <= threshold]
            # 計算新長度並寫入
            count = len(filtered)
            new_line = f"{count} " + " ".join(filtered)
            outfile.write(new_line + "\n")

# Usage
input_path = "train.txt"
output_path = "citations.txt"
process_dataset(input_path, output_path)

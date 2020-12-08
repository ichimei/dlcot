import random
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_num", type=int, help="number of train")
    parser.add_argument("val_num", type=int, help="number of val")
    parser.add_argument("-f", "--files-path" , help="Path to files", required=True)
    parser.add_argument("-t", "--train-out" , help="Path to train list")
    parser.add_argument("-v", "--val-out", help="Path to val list")
    parser.add_argument("-a", "--train-val-out", help="Path to trainval list")
    parser.add_argument("-r", "--random", action='store_true', help="Path to trainval list")
    args = parser.parse_args()
    print(args)
    return args

def split_train_val(files, train_num, val_num, train_out, val_out, train_val_out):
    full_num = len(files)
    tv_num = train_num + val_num
    if tv_num > full_num:
        raise Exception(f'Has {full_num} but given {tv_num}')
    test_num = full_num - tv_num
    print(f'[INFO] Has {full_num}, train {train_num}, val {val_num}, test {test_num}')
    trains = files[:train_num]
    vals = files[train_num:tv_num]
    trains_vals = files[:tv_num]
    pairs = [(trains, train_out), (vals, val_out), (trains_vals, train_val_out)]
    for fs, fout in pairs:
        if fout is not None:
            with open(fout, 'w') as f:
                for fline in fs:
                    f.write(fline + '\n')

def get_files(path, rand):
    files = os.listdir(path)
    files = [f.rsplit('.', maxsplit=1)[0] for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if rand:
        random.shuffle(files)
    else:
        files.sort()
    return files

def main():
    args = parse_args()
    train_num = args.train_num
    val_num = args.val_num
    files = get_files(args.files_path, rand=args.random)
    train_out = args.train_out
    val_out = args.val_out
    train_val_out = args.train_val_out
    split_train_val(files, train_num, val_num, train_out, val_out, train_val_out)

if __name__ == '__main__':
    main()

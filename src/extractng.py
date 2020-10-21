# -*- coding: utf-8 -*-
# +
ng = 0
num_line = 1

ng_file = '../ng.txt'
import glob
files = glob.glob('../lg_train/*.txt')
data = None
for idx, file in enumerate(files):
    print(idx + 1, "번째 파일 ", file, "가져오는 중.")
    f = open(file, mode='r')
    f_ng = open(ng_file, mode='a')
    while True:
        line = f.readline()
        if not line: break
        # print(num_line)
        arr = line.split('\t')
        if arr[0] == '1':
            f_ng.write(line)
            ng += 1
            print(num_line, "번째 라인에서 총 ", ng, '번째 ng 확인.')
        num_line += 1

    f.close()

print("총 ", num_line, "개 데이터 중 총 ", ng, "개 NG 확인.")

# -





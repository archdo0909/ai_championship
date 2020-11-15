
import os


# (1) 다운로드한 데이터(폴더)에서 정상과 불량 데이터 분리
def split_data(data_dir):
    normal_data_fpath = os.path.join('/workspace/demon', 'normal.txt')
    abnormal_data_fpath = os.path.join('workspace/demon', 'abnormal.txt')

    data_fpaths = [fname for fname in os.listdir(data_dir) if fname.endswith('.txt')]
    for data_fpath in data_fpaths:
        with open(data_fpath, 'r') as f:
            for line in f.readlines():
                print(line)
                break

        with open(normal_data_fpath, 'a') as nf:
            nf.write()
        with open(abonormal_data_fpath, 'a') as af:
            af.write()


# (2) 위에서 저장된 데이터에 대하여 데이터 전처리 실행 및 저장
def preprocess_data():
    pass

# (3) 전처리된 데이터에 대하여 pretrained deep ensemble 모델로 prediction 실행
def make_prediction():
    pass


if __name__ == '__main__':
    # e.g. /workspace/demon/testdir
    split_data('/workspace/lg_train_test')
    

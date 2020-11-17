
import numpy as np

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = []
    with open('/workspace/peter/sampled/sampled_random700.txt', "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            data.append(line.strip().split("\t"))

    data = np.array(data)
    X = data[:, 1:]
    y = np.int32(data[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    np.savetxt('train_data.txt', X)
    np.savetxt('test_data.txt', y)
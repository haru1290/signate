import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


def main():
    train = pd.read_csv('../data/train.csv', header=0)
    test = pd.read_csv('../data/test.csv', header=0)


if __name__ == '__main__':
    main()
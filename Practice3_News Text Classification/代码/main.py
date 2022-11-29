# ==================================
# import
# ==================================
import time
import torch
import numpy as np
from train_test_eval import train, init_net_params, test
from Configuration import Config
from TextRNN import Model
from utils import build_dataset, build_iterator, DatasetIterator, get_time_dif
import os

if __name__ == "__main__":
    config = Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True   # make sure the same outcome

    start_time = time.time()
    print("Loading Data........")
    train_data, test_data = build_dataset(config)
    train_iter, test_iter = build_iterator(train_data, config), build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Loading Uses Time: {}".format(time_dif))

    # train
    model = Model(config).to(config.device)
    start_time = time.time()
    init_net_params(model)
    # print(model.parameters())
    if os.path.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path))
        model.train()
    model = train(config, model, train_iter)
    test(config, model, test_iter)

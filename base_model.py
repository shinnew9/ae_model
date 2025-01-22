import pickle
import os
import scipy
import torch
import numpy as np
import json
import pandas as pd
import shutil
from pathlib import Path

class BaseModel(object):
    def __init__(self, args):
        self.args = args
        print("Selected GPU ID: {}".format(args.gpu_id))
        self.device = torch.device("cuda" if args.cuda else "cpu", args.gpu_id[0])
        print(self.device)
        
        # 원본에서 try~except 없앰
        self.epoch = 0
        self.model = self.init_model()

        self.checkpoint_dir = f"models/checkpoint/{args.export_dir}"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = f"{self.checkpoint_dir}/checkpoint.tar"

        self.log_path = Path(f"logs/{args.export_dir}/log.csv")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_data = self.get_log_header()
        if not os.path.exists(self.checkpoint_path):
            np.savetxt(self.log_path, [log_data], fmt="%s")

    def init_model(self):
        pass  # 모델 초기화는 상속받은 클래스에서 구현


    def get_log_header(self):
        return "loss,loss_var,time"


    def save_checkpoint(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at {self.checkpoint_path}")


    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            print(f"Checkpoint loaded: Epoch {self.epoch}")
            return self.epoch, checkpoint['loss']
        else:
            print(f"No checkpoint found at {self.checkpoint_path}")
            return None


    def fit_anomaly_score_distribution(self, y_pred, score_distr_file_path):
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        with open(score_distr_file_path, "wb") as f:
            pickle.dump(gamma_params, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Fitted Gamma distribution saved at {score_distr_file_path}")


    def calc_decision_threshold(self, score_distr_file_path, percentile=90):
        with open(score_distr_file_path, "rb") as f:
            shape_hat, loc_hat, scale_hat = pickle.load(f)
        threshold = scipy.stats.gamma.ppf(q=percentile / 100, a=shape_hat, loc=loc_hat, scale=scale_hat)
        print(f"Threshold calculated: {threshold}")
        return threshold

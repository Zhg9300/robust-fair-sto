
import os

from components.main import initialize, read_params, outFunc

from components.Algorithm import Algorithm
from components.Client import Client
from components.DataLoader import DataLoader
from components.Module import Module
from components.Metric import Metric
from components.seed import setup_seed

from components.metric.Correct import Correct
from components.model.CNN import CNN
from components.model.LinearRegression import LinearRegression
from components.model.MLP import MLP


from components.algorithm.FedAvg.FedAvg import FedAvg
from components.algorithm.qFedAvg.qFedAvg import qFedAvg
from components.algorithm.AFL.AFL import AFL
from components.algorithm.FedFV.FedFV import FedFV
from components.algorithm.FedMGDA_plus.FedMGDA_plus import FedMGDA_plus
from components.algorithm.DRFL.DRFL import DRFL
from components.algorithm.AdaFed.AdaFed import AdaFed

from components.dataloaders.indexed_data import IndexedBatchData
from components.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from components.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from components.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from components.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir
from components.dataloaders.DataLoader_linear_regression import DataLoader_linear_regression


data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)

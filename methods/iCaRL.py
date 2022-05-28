from matplotlib.transforms import Transform
import numpy
from torch import log
from utils.train_utils import select_model, select_optimizer
from utils.data_loader import ImageDataset
import random
from torch.utils.data import DataLoader
import pandas as pd
import logging
import torch.nn as nn
import torch
from torch.nn import functional as F

from methods.finetune import Finetune

logger = logging.getLogger()

class iCaRL(Finetune):
    def __init__(self, criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs):
        super(iCaRL, self).__init__(criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs)
        # some data needed by iCaRL

        # restore the current exemplar set
        self.exemplar_set = {}
        # restore the current classes mean feature
        self.exemplar_mean = {}
    
    def set_current_dataset(self, train_datalist, test_datalist):
        super(iCaRL, self).set_current_dataset(train_datalist, test_datalist)

    # set_current_dataset -> use the Finetune's
    # get_dataloader      -> use the Finetune's
    # before_task         -> use the Finetune's
    # evaluation          -> use the Finetune's


    # todo change the train algorithm
    def train(self, cur_iter):
        logger.info("#" * 10 + "Start Training" + "#" * 10)
        # get the train_list
        train_list = self.combine_exemplar_and_current()    
        test_list  = self.test_list

        train_loader, test_loader = self.get_dataloader(
            self.batch_size, self.n_woker, train_list, test_list
        )

        logger.info(f"New training samples: {len(self.train_list)}")
        logger.info(f"In-memory samples: {len(train_list) - len(self.train_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        best_acc = 0.0
        eval_dict = dict()
        n_batches = len(train_loader)
        for epoch in range(self.n_epoch):
            if epoch > 0:
                self.scheduler.step()
            
            total_loss, correct, num_data = 0.0, 0.0, 0.0
            self.model.train()
            for i, data in enumerate(train_loader):
                x = data['image'].to(self.device)
                y = data['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                logit = self.model(x)
                loss = self.criterion(logit, y)
                preds = torch.argmax(logit, dim=-1)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y).item()
                num_data += y.size(0)
            
            eval_dict = self.evaluation(test_loader=test_loader, criterion=self.criterion)

            cls_acc = "cls_acc: ["
            for _ in eval_dict['cls_acc']:
                cls_acc += format(_, '.3f') + ', '
            cls_acc += ']'

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{self.n_epoch} | lr {self.optimizer.param_groups[0]['lr']:.4f} | train_loss {total_loss/n_batches:.4f} | train_acc {correct/num_data:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} |"
            )

            # 输出每个类别的准确率，但是cifar100类别数目一多，输出就太乱了，取决你们自己
            #logger.info(cls_acc)


            best_acc = max(best_acc, eval_dict["avg_acc"])
            
        return best_acc, eval_dict


    def after_task(self, cur_iter):
        # update the num_learned_class
        self.num_learned_class = self.num_learning_class
        # the size which each exemplar_set should be
        k = self.memory_size // self.num_learned_class

        ## first
        ## reduce the size of current exemplar_set
        logger.info("#"*15 + " reduce exemplar set " + "#"*15)
        self.reduce_exemplar_set(k)

        ## second
        ## construct the new exemplar_set
        logger.info("#"*15 + " construct exemplar set " + "#"*15)
        self.construct_exemplar_set(cur_iter, k)        

        ## third
        ## caculate the mean feature representation for every class
        logger.info("#"*15 + " caculate exemplar mean " + "#"*15)
        self.caculate_exemplar_mean()

        ## finally
        ## use the exemplar_mean to classify the test data, and compute the accuracy
        logger.info("#"*10 + " evaluate using nearest mean exempalrs classification " + "#"*10)
        self.nearest_mean_exemplars_classify()


    
    ## Some helper function

    # return the datalist struct which combined the exemplar and new datalist
    def combine_exemplar_and_current(self):
        datalist = []
        # datalist.append({'file_name': img_path, 'label': start_label})
        ## first, add all exemplar
        for key, value in self.exemplar_set.items():
            for img_path in value:
                datalist.append({'file_name':img_path, 'label':key})
        ## second, add all new data
        datalist = datalist + self.train_list
        return datalist 


    ## just keep k images started from the exemplar
    ## so we need to keep the exemplar_set sorted by the distance from images to the class mean
    ## when we construct it
    def reduce_exemplar_set(self, k):
        for key, value in self.exemplar_set.items():
            self.exemplar_set[key] = value[:k]

    def construct_exemplar_set(self, cur_iter, k):
        ## every iteration, we train 20 new classes
        new_img_dataset = ImageDataset (
            pd.DataFrame(self.train_list),
            self.dataset,
            self.train_transform
        )
        for label in range(cur_iter * 20, cur_iter * 20 + 20):
            img_set = ImageDataset(
                new_img_dataset.get_image_class(label),
                self.dataset,
                self.train_transform
            )
            ## select k imgs from the data set
            res = self.compute_k_nearest_img(img_set, k)
            ## add to the exemplar set
            self.exemplar_set[label] = res

        
    def caculate_exemplar_mean(self):
        self.exemplar_mean = {}
        for label in range(len(self.exemplar_set)):
            img_dataset = self.exemplar_to_dataset(label)
            class_mean, _ = self.compute_the_class(img_dataset.to_tensor())
            self.exemplar_mean[label] = class_mean


    def nearest_mean_exemplars_classify(self):
        correct = 0
        total = 0

        test_dataset = ImageDataset(
            pd.DataFrame(self.test_list),
            self.dataset,
            self.test_transform
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.n_woker,
        )
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data['image'].to(self.device)
                labels = data['label'].to(self.device)
                preds = self.classify(x)
                correct += (labels.cpu() == preds.cpu()).sum()
                total += len(labels)
        logger.info(f"total accuracy:{(correct * 100 / total):.3f}%")


    def compute_k_nearest_img(self, dataset : ImageDataset, k):
        ## compute the mean of the current class
        ## and extract the feature for every imgs
        data = dataset.to_tensor()
        class_mean, feature_output = self.compute_the_class(data)

        exemplar_list = []
        exemplar_sum = numpy.zeros((1, 512))
        
        ## select k imgs from dataset
        for i in range(k):
            ## compute the distance from each imgs to the class mean
            x = class_mean - (exemplar_sum + feature_output)/(i + 1)
            x = numpy.linalg.norm(x, axis=1)
            ## add the img into the exemplar
            idx = numpy.argmin(x)
            exemplar_sum += feature_output[idx]
            exemplar_list.append(dataset[idx]['image_name'])

        return exemplar_list

    ## compute the mean of the class
    ## exetract the feature of each image
    def compute_the_class(self, data : torch.Tensor):
        
        ## copy the data to GPU
        data = data.to(self.device)
        ## feature [len(data), 512], the output feature is 512 dimension
        features = F.normalize(self.model.featrue_extractor(data).detach()).cpu().numpy()
        class_mean = numpy.mean(features, axis=0)
        return class_mean, features

    ## return the dataset of exemplar label
    def exemplar_to_dataset(self, label):
        datalist = []
        exemplar_list = self.exemplar_set[label]
        for img_path in exemplar_list:
            datalist.append({'file_name':img_path, 'label':label})
        img_dataset = ImageDataset(
            pd.DataFrame(datalist),
            self.dataset,
            self.train_transform
        )
        return img_dataset

    def classify(self, x):
        res = []
        _x = F.normalize(self.model.featrue_extractor(x).detach()).cpu().numpy()
        exemplar_means = [value for _, value in self.exemplar_mean.items()]
        exemplar_means = numpy.array(exemplar_means)
        for input in _x:
            dif = input - exemplar_means
            dist = numpy.linalg.norm(dif, ord=2, axis=1)
            label = numpy.argmin(dist)
            res.append(label)
        return torch.tensor(res)





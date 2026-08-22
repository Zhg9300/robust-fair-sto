import copy

import numpy as np
import torch

from components.param_utils import build_optimizer_like


class Client:
    def __init__(self, id=None, module=None, device=None,
                 train_setting=None, metric_list=None):
        self.id = id
        self.module = copy.deepcopy(module)
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.test_loss_evaluator = None
        self.training_batch_num = 0
        self.physical_training_batch_num = 0
        self.test_batch_num = 0
        self.sgd_step = train_setting["sgd_step"]
        self.micro_batch_size = int(train_setting.get("micro_batch_size", 0))
        self.metric_history = {
            "training_loss": [],
            "test_loss": [],
            "local_test_number": 0,
        }
        for metric in self.metric_list:
            self.metric_history[metric.name] = []
            if metric.name == "correct":
                self.metric_history["test_accuracy"] = []
        self.criterion = train_setting["criterion"].to(self.device)

    def update_data(self, id, local_training_data, local_training_number,
                    local_test_data, local_test_number, test_loss_evaluator=None):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.test_loss_evaluator = test_loss_evaluator
        self.training_batch_num = len(local_training_data)
        physical_count = getattr(local_training_data, "physical_batch_count", None)
        self.physical_training_batch_num = (
            physical_count(self.micro_batch_size)
            if callable(physical_count)
            else self.training_batch_num
        )
        self.test_batch_num = 0 if local_test_data is None else len(local_test_data)

    def _to_model_device(self, inputs, targets):
        if inputs.device != self.device:
            inputs = inputs.to(self.device, non_blocking=True)
        if targets.device != self.device:
            targets = targets.to(self.device, non_blocking=True)
        return inputs, targets

    def _training_batches(self, physical_batch_size=0):
        iterator = getattr(self.local_training_data, "iter_batches", None)
        if callable(iterator):
            return iterator(physical_batch_size)
        return iter(self.local_training_data)

    @property
    def _is_full_batch(self):
        return getattr(self.local_training_data, "batch_size", None) == "full"

    @staticmethod
    def _poison_labels(labels, label_mapping):
        if label_mapping is None:
            return labels
        integer_dtypes = {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }
        if labels.dtype not in integer_dtypes:
            raise ValueError("Label poisoning requires integer training labels.")
        mapping = torch.as_tensor(
            label_mapping, dtype=torch.long, device=labels.device
        )
        if mapping.ndim != 1 or mapping.numel() != 10:
            raise ValueError("Label poisoning mapping must contain exactly 10 labels.")
        if torch.any((mapping < 0) | (mapping > 9)):
            raise ValueError("Label poisoning mapping values must be in [0, 9].")
        if labels.numel() and torch.any((labels < 0) | (labels > 9)):
            raise ValueError("Label poisoning requires training labels in [0, 9].")
        return mapping[labels.long()]

    def cal_gradient_loss(self, label_mapping=None):
        self.module.model.train()
        self.module.model.zero_grad(set_to_none=True)
        total_loss = 0.0
        physical_size = self.micro_batch_size if self._is_full_batch else 0
        for inputs, targets in self._training_batches(physical_size):
            inputs, targets = self._to_model_device(inputs, targets)
            poisoned_targets = self._poison_labels(targets, label_mapping)
            loss = self.criterion(self.module.model(inputs), poisoned_targets)
            total_loss += loss.item() * targets.size(0)
            (loss * (targets.size(0) / self.local_training_number)).backward()
        return (
            self.module.span_model_grad_to_vec(),
            float(total_loss / self.local_training_number),
        )

    def cal_full_loss(self, label_mapping=None):
        """Evaluate the current model loss over all local training samples."""
        self.module.model.eval()
        total_loss = 0.0
        physical_size = self.micro_batch_size if self._is_full_batch else 0
        with torch.no_grad():
            for inputs, targets in self._training_batches(physical_size):
                inputs, targets = self._to_model_device(inputs, targets)
                poisoned_targets = self._poison_labels(targets, label_mapping)
                loss = self.criterion(self.module.model(inputs), poisoned_targets)
                total_loss += loss.item() * targets.size(0)
        return float(total_loss / self.local_training_number)

    def cal_gradient_loss_sgd(self, label_mapping=None):
        self.module.model.train()
        sample_index = int(np.random.choice(len(self.local_training_data)))
        inputs, targets = self.local_training_data[sample_index]
        inputs, targets = self._to_model_device(inputs, targets)
        poisoned_targets = self._poison_labels(targets, label_mapping)
        self.module.model.zero_grad(set_to_none=True)
        loss = self.criterion(self.module.model(inputs), poisoned_targets)
        loss.backward()
        gradient = self.module.span_model_grad_to_vec()
        self.module.model.zero_grad(set_to_none=True)
        return gradient, loss.item()

    def evaluate_gradient(self, label_mapping=None, use_full_loss=False):
        if self.sgd_step:
            full_loss = (
                self.cal_full_loss(label_mapping)
                if use_full_loss
                else None
            )
            gradient, batch_loss = self.cal_gradient_loss_sgd(label_mapping)
            return gradient, full_loss if use_full_loss else batch_loss
        return self.cal_gradient_loss(label_mapping)

    def train_local(self, epochs, lr, label_mapping=None):
        if self.sgd_step:
            loss = self._train_sgd(epochs, lr, label_mapping)
        else:
            loss = self._train(epochs, lr, label_mapping)
        return self.module.span_model_params_to_vec(), loss

    def _train_sgd(self, epochs, lr, label_mapping):
        optimizer = build_optimizer_like(
            self.train_setting["optimizer"],
            filter(lambda parameter: parameter.requires_grad,
                   self.module.model.parameters()),
            lr=lr,
        )
        self.module.model.train()
        loss_value = None
        for _ in range(epochs):
            sample_index = int(np.random.choice(len(self.local_training_data)))
            inputs, targets = self.local_training_data[sample_index]
            inputs, targets = self._to_model_device(inputs, targets)
            poisoned_targets = self._poison_labels(targets, label_mapping)
            optimizer.zero_grad()
            loss = self.criterion(self.module.model(inputs), poisoned_targets)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
        return loss_value

    def _train(self, epochs, lr, label_mapping):
        optimizer = build_optimizer_like(
            self.train_setting["optimizer"],
            filter(lambda parameter: parameter.requires_grad,
                   self.module.model.parameters()),
            lr=lr,
        )
        self.module.model.train()
        average_loss = None
        for _ in range(epochs):
            total_loss = 0.0
            if self._is_full_batch:
                optimizer.zero_grad()
                batches = self._training_batches(self.micro_batch_size)
            else:
                batches = iter(self.local_training_data)
            for inputs, targets in batches:
                inputs, targets = self._to_model_device(inputs, targets)
                poisoned_targets = self._poison_labels(targets, label_mapping)
                if not self._is_full_batch:
                    optimizer.zero_grad()
                loss = self.criterion(self.module.model(inputs), poisoned_targets)
                if self._is_full_batch:
                    (loss * (targets.size(0) / self.local_training_number)).backward()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * targets.size(0)
            if self._is_full_batch:
                optimizer.step()
            average_loss = total_loss / self.local_training_number
            self.metric_history["training_loss"].append(average_loss)
        return average_loss

    def test(self):
        if callable(self.test_loss_evaluator):
            self.metric_history["local_test_number"] = 1
            self.metric_history["test_loss"].append(
                float(self.test_loss_evaluator(self.module))
            )
            return copy.deepcopy(self.metric_history)

        metric_values = {"test_loss": 0.0}
        metric_values.update({metric.name: 0.0 for metric in self.metric_list})
        self.module.model.eval()
        with torch.no_grad():
            self.metric_history["local_test_number"] = self.local_test_number
            iterator = getattr(self.local_test_data, "iter_batches", None)
            batches = (
                iterator(self.micro_batch_size)
                if callable(iterator) and self._is_full_batch and self.micro_batch_size
                else iter(self.local_test_data)
            )
            for inputs, targets in batches:
                inputs, targets = self._to_model_device(inputs, targets)
                output = self.module.model(inputs)
                loss = self.criterion(output, targets)
                metric_values["test_loss"] += loss.item() * targets.size(0)
                for metric in self.metric_list:
                    metric_values[metric.name] += metric.calc(output, targets)
        self.metric_history["test_loss"].append(
            metric_values["test_loss"] / self.local_test_number
        )
        for metric in self.metric_list:
            self.metric_history[metric.name].append(metric_values[metric.name])
            if metric.name == "correct":
                self.metric_history["test_accuracy"].append(
                    100 * metric_values["correct"] / self.local_test_number
                )
        return copy.deepcopy(self.metric_history)

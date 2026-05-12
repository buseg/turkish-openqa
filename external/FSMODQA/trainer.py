import copy
import glob
import json
import logging
import math
import os
import random
import re
import shutil
from collections import Counter
from itertools import chain
from statistics import mean
from typing import Dict, List, Tuple, Optional, Any, Union, Mapping

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule, PreTrainedModel

from dataloader import GenericDataLoader, EncodeDataset, EncodeCollator, ReaderDataset, ReaderCollator
from evals.eval_xor_retrieve import read_jsonlines, evaluate_top_k_hit
from model import RRForConditionalGeneration
from retriever import FaissIPRetriever, write_ranking
from utils import AverageMeter, ProgressMeter, RandContext, compute_colbert_scores

logger = logging.getLogger(__name__)


class ReaderTrainer(object):
    def __init__(self, model, train_dataset, data_collator, training_args, data_args, tokenizer):
        super(ReaderTrainer, self).__init__()
        self.training_args = training_args
        self.data_args = data_args
        self.args = training_args
        self.config = model.config
        self.epoch = 0

        if dist.is_initialized() and dist.get_world_size() > 1:
            assert self.training_args.negatives_x_device, self.training_args.negatives_x_device
        self._dist_loss_scale_factor = dist.get_world_size() if self.training_args.negatives_x_device else 1

        if self.training_args.negatives_x_device:
            assert dist.is_initialized() and dist.get_world_size() > 1, \
                ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator

        if isinstance(self.train_dataset, list):
            self.train_dataloader = []
            for idx, dataset in enumerate(train_dataset):
                self.train_dataloader.append(self.get_train_dataloader(dataset))
        else:
            self.train_dataloader = self.get_train_dataloader(self.train_dataset)

        if isinstance(self.train_dataloader, list):
            assert training_args.multi_task, \
                ValueError('can only have multiple datasets when using multi-task learning')
            self.num_training_steps = 0
            for dataloader in self.train_dataloader:
                self.num_training_steps += len(dataloader) // self.training_args.gradient_accumulation_steps
        else:
            self.num_training_steps = len(self.train_dataloader) // self.training_args.gradient_accumulation_steps
        if self.training_args.max_steps > 0:
            self.max_step = self.training_args.max_steps
            self.num_train_epochs = 1
        else:
            self.max_step = self.training_args.num_train_epochs * self.num_training_steps
            self.num_train_epochs = math.ceil(self.training_args.num_train_epochs)

        if training_args.gradient_checkpointing:
            model.encoder.gradient_checkpointing = training_args.gradient_checkpointing
        self.model = self.setup_model(model)
        self.optimizer = self.get_optimizer(self.model, self.training_args.weight_decay,
                                            self.training_args.learning_rate)
        if self.training_args.scheduler_type == "linear_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.training_args.warmup_ratio * self.max_step,
                num_training_steps=self.max_step
            )
        else:
            assert self.training_args.scheduler_type == "constant", self.training_args.sheduler_type
            self.scheduler = get_constant_schedule(self.optimizer)

        os.makedirs(self.training_args.output_dir, exist_ok=True)
        self.tb_writer = None
        if self.is_world_process_zero():
            tb_log_dir = self.training_args.tensorboard_log_dir
            if tb_log_dir is None:
                tb_log_dir = os.path.join(self.training_args.output_dir, "runs")
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(tb_log_dir)
                logger.info("Writing TensorBoard logs to %s", tb_log_dir)
            except ImportError:
                logger.warning("TensorBoard is not installed; skipping TensorBoard logging.")

        self.use_amp = False
        self.amp_dtype = None
        self.scaler = None
        if self.training_args.fp16 or self.training_args.bf16:
            self.use_amp = True
            self.amp_dtype = torch.float16 if self.training_args.fp16 else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler()

    def setup_model(self, model):
        model = model.to(self.training_args.device)
        if self.training_args.n_gpu > 1:
            model = nn.DataParallel(model)
        elif self.training_args.local_rank != -1:
            kwargs = {}
            if self.training_args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.training_args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.training_args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.training_args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.training_args.device] if self.training_args.n_gpu != 0 else None,
                output_device=self.training_args.device if self.training_args.n_gpu != 0 else None,
                broadcast_buffers=False,
                **kwargs,
            )
        return model

    def get_optimizer(self, model, weight_decay, lr):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        adam_kwargs = {
            "betas": (self.training_args.adam_beta1, self.training_args.adam_beta2),
            "eps": self.training_args.adam_epsilon,
        }
        return AdamW(optimizer_grouped_parameters, lr=lr, **adam_kwargs)

    def _save(self, model_to_save, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        if isinstance(model_to_save, PreTrainedModel):
            model_to_save.save_pretrained(output_dir)
        else:
            model_to_save.save(output_dir)

    def save_model(self):
        if self.is_world_process_zero():
            self._save(self.model)
        if self.tb_writer is not None:
            self.tb_writer.close()

    def is_world_process_zero(self) -> bool:
        return self.training_args.process_index == 0

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.training_args.device)
        return data

    def _prepare_inputs(
            self,
            inputs: Union[Tuple[Dict[str, Union[torch.Tensor, Any]], ...], Dict]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        if isinstance(inputs, Mapping):
            return self._prepare_input(inputs)
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.training_args.device))
            else:
                prepared.append(self._prepare_input(x))
        return prepared

    def _log_tensorboard_train(self, global_step, loss, avg_loss=None):
        if self.tb_writer is None or global_step == 0:
            return
        self.tb_writer.add_scalar("train/loss", loss, global_step)
        if avg_loss is not None:
            self.tb_writer.add_scalar("train/loss_avg", avg_loss, global_step)
        self.tb_writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], global_step)
        self.tb_writer.flush()

    def _tensorboard_generation_examples(self):
        dataset = self.train_dataset[0] if isinstance(self.train_dataset, list) else self.train_dataset
        data_dir = self.data_args.train_dir
        eval_path = os.path.join(data_dir, "validation.jsonl")
        eval_query_file = "validation.query.jsonl"
        if not os.path.exists(eval_path):
            eval_path = os.path.join(data_dir, self.data_args.train_path)
            eval_query_file = self.data_args.query_file

        sample_size = self.training_args.tb_metric_examples or self.training_args.tb_log_examples
        examples = getattr(self, "_tb_generation_examples_cache", None)
        if examples is None:
            with open(eval_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
            if len(examples) > sample_size:
                rng = random.Random(self.training_args.seed)
                examples = rng.sample(examples, sample_size)
            self._tb_generation_examples_cache = examples

        queries = GenericDataLoader(
            data_dir,
            corpus_file=self.data_args.corpus_file,
            query_file=eval_query_file,
        ).load_queries()
        self._tb_query_cache = queries

        sample_dataset = ReaderDataset(
            queries=queries,
            corpus=dataset.corpus,
            tokenizer=self.tokenizer,
            train_path=examples,
            data_args=self.data_args,
            eval_mode=True,
        )
        sample_collator = ReaderCollator(
            self.tokenizer,
            max_query_length=self.data_args.max_query_length,
            max_passage_length=self.data_args.max_passage_length,
            max_query_passage_length=self.data_args.max_query_passage_length,
            max_answer_length=self.data_args.max_answer_length,
            separate_joint_encoding=self.training_args.separate_joint_encoding,
        )
        return DataLoader(
            sample_dataset,
            batch_size=1,
            sampler=SequentialSampler(sample_dataset),
            collate_fn=sample_collator,
            drop_last=False,
            num_workers=0,
        )

    def _reader_validation_sample_scores(self, global_step=None, log_text=False):
        if self.training_args.tb_metric_examples <= 0 and self.training_args.tb_log_examples <= 0:
            return None
        model = self.model
        while hasattr(model, 'module'):
            model = model.module

        was_training = model.training
        model.eval()
        rows = ["| qid | question | gold | prediction |", "| --- | --- | --- | --- |"]
        text_examples_remaining = self.training_args.tb_log_examples if log_text else 0
        old_n_passages = model.n_passages
        model.n_passages = self.data_args.train_n_passages
        em_scores = []
        f1_scores = []

        try:
            with torch.no_grad():
                for batch in self._tensorboard_generation_examples():
                    if self.training_args.separate_joint_encoding:
                        qids, reader_inputs, batch_answer, _ = batch
                        bsz, seq_len = reader_inputs["input_ids"].size()
                        input_ids = reader_inputs["input_ids"].view(
                            bsz // self.data_args.train_n_passages,
                            self.data_args.train_n_passages,
                            seq_len,
                        )
                        model_inputs = {
                            "input_ids": input_ids.to(self.training_args.device),
                            "attention_mask": reader_inputs["attention_mask"].to(self.training_args.device),
                            "independent_mask": reader_inputs["independent_mask"].to(self.training_args.device),
                        }
                    else:
                        qids, _, _, reader_inputs, batch_answer, _ = batch
                        bsz, seq_len = reader_inputs["input_ids"].size()
                        input_ids = reader_inputs["input_ids"].view(
                            bsz // self.data_args.train_n_passages,
                            self.data_args.train_n_passages,
                            seq_len,
                        )
                        model_inputs = {
                            "input_ids": input_ids.to(self.training_args.device),
                            "attention_mask": reader_inputs["attention_mask"].to(self.training_args.device),
                        }

                    outputs = model.generate(
                        **model_inputs,
                        max_length=self.data_args.max_answer_length,
                        num_beams=1,
                    )
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    qid = qids[0]
                    query = self._tensorboard_query_text(qid)
                    gold = batch_answer[0]
                    gold_answers = self._tensorboard_answers(qid, gold)
                    em_scores.append(max(self._metric_exact_match(prediction, answer) for answer in gold_answers))
                    f1_scores.append(max(self._metric_f1(prediction, answer) for answer in gold_answers))
                    if text_examples_remaining > 0:
                        rows.append(
                            f"| {self._tensorboard_escape(qid)} | {self._tensorboard_escape(query)} | "
                            f"{self._tensorboard_escape(gold)} | {self._tensorboard_escape(prediction)} |"
                        )
                        text_examples_remaining -= 1

            scores = {
                "em": mean(em_scores) * 100 if em_scores else 0.0,
                "f1": mean(f1_scores) * 100 if f1_scores else 0.0,
            }
            if self.tb_writer is not None and global_step is not None:
                if log_text:
                    self.tb_writer.add_text("eval/example_generations", "\n".join(rows), global_step)
                self.tb_writer.add_scalar("eval_sample/em", scores["em"], global_step)
                self.tb_writer.add_scalar("eval_sample/f1", scores["f1"], global_step)
                self.tb_writer.flush()
            return scores
        finally:
            model.n_passages = old_n_passages
            if was_training:
                model.train()

    def _log_tensorboard_generations(self, global_step):
        if self.tb_writer is None:
            return
        if self.training_args.tb_log_generation_steps <= 0:
            return
        if global_step == 0 or global_step % self.training_args.tb_log_generation_steps != 0:
            return
        self._reader_validation_sample_scores(global_step, log_text=self.training_args.tb_log_examples > 0)

    @staticmethod
    def _normalize_metric_text(text):
        text = str(text).casefold()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    @classmethod
    def _metric_tokens(cls, text):
        return re.findall(r"\w+", cls._normalize_metric_text(text), flags=re.UNICODE)

    @classmethod
    def _metric_exact_match(cls, prediction, answer):
        return float(cls._normalize_metric_text(prediction) == cls._normalize_metric_text(answer))

    @classmethod
    def _metric_f1(cls, prediction, answer):
        pred_tokens = cls._metric_tokens(prediction)
        answer_tokens = cls._metric_tokens(answer)
        if not pred_tokens or not answer_tokens:
            return float(pred_tokens == answer_tokens)
        common = Counter(pred_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(answer_tokens)
        return 2 * precision * recall / (precision + recall)

    def _tensorboard_answers(self, qid, fallback):
        queries = getattr(self, "_tb_query_cache", None)
        if queries is not None and qid in queries:
            query = queries[qid]
            if isinstance(query, tuple) and len(query) >= 2:
                answers = query[1]
                if isinstance(answers, list) and answers:
                    return [str(answer) for answer in answers]
        return [str(fallback)]

    def _tensorboard_query_text(self, qid):
        queries = getattr(self, "_tb_query_cache", None)
        if queries is None:
            dataset = self.train_dataset[0] if isinstance(self.train_dataset, list) else self.train_dataset
            queries = dataset.queries
        query = queries.get(qid, "")
        if isinstance(query, tuple):
            query = query[0]
        if isinstance(query, list):
            query = query[0]
        return str(query)

    def _tensorboard_escape(self, text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    def get_train_dataloader(self, train_dataset) -> DataLoader:
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.training_args.world_size > 1:
            seed = self.training_args.data_seed if self.training_args.data_seed is not None else self.training_args.seed
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.training_args.world_size,
                rank=self.training_args.process_index,
                seed=seed,
            )
        else:
            train_sampler = RandomSampler(train_dataset)

        train_batch_size = self.training_args.train_batch_size

        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

    def compute_loss(self, inputs, global_step=None):
        _, query, passage, reader_inputs, _, _ = inputs

        if global_step < self.training_args.distillation_start_steps:
            outputs = self.model(**reader_inputs, return_dict=True)
            return outputs.loss

        self.model.eval()
        with torch.no_grad():
            self.model.requires_grad_(False)
            decoder_input_ids = torch.ones(reader_inputs['labels'].size(0), 1, dtype=torch.long,
                                           device=self.training_args.device) * self.config.decoder_start_token_id
            outputs = self.model(
                input_ids=reader_inputs['input_ids'],
                attention_mask=reader_inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                return_dict=True,
            )
            self.model.requires_grad_(True)
        cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
        bsz, n_heads, _ = cross_attention_scores.size()
        scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
        decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
        self.model.train()

        outputs = self.model(**reader_inputs, query=query, passage=passage, return_dict=True)
        reader_loss = outputs.loss

        query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
        if self.training_args.negatives_x_device:
            query_vector = self.dist_gather_tensor(query_vector)
            passage_vector = self.dist_gather_tensor(passage_vector)
            decoder_scores = self.dist_gather_tensor(decoder_scores)
        retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        assert decoder_scores.size()[0] == retriever_scores.size()[0], (decoder_scores.size(), retriever_scores.size())
        teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
        bsz = decoder_scores.size()[0]
        for i in range(bsz):
            start = i * self.data_args.train_n_passages
            end = (i + 1) * self.data_args.train_n_passages
            teacher_scores[i, start:end] = decoder_scores[i]

        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')
        loss = reader_loss + retriever_loss * self._dist_loss_scale_factor

        return loss / self.training_args.gradient_accumulation_steps

    def grad_cache_compute_loss(self, inputs, global_step=None):
        _, query, passage, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, labels = reader_inputs['input_ids'], reader_inputs['attention_mask'], \
                                             reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        reader_loss = 0
        for idx, (input_ids, attention_mask, labels) in enumerate(
                zip(input_ids_chunks, attention_mask_chunks, labels_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    ).loss
                    loss /= self.training_args.gc_chunk_size
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                return loss

            if idx != len(labels_chunks) - 1:
                with self.model.no_sync():
                    loss = chunk_forward()
            else:
                loss = chunk_forward()
            reader_loss = reader_loss + loss

        if global_step < self.training_args.distillation_start_steps:
            return reader_loss

        self.model.eval()
        with torch.no_grad():
            self.model.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                decoder_scores_list = []
                for input_ids, attention_mask, labels in zip(input_ids_chunks, attention_mask_chunks, labels_chunks):
                    decoder_input_ids = torch.ones(labels.size(0), 1, dtype=torch.long,
                                                   device=self.training_args.device) * self.config.decoder_start_token_id
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )
                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    decoder_scores = scores.sum(dim=-1).mean(dim=1).detach()
                    decoder_scores_list.append(decoder_scores)
                decoder_scores = torch.cat(decoder_scores_list, dim=0)
            self.model.requires_grad_(True)
        self.model.train()

        def compute_vector(inputs_ids_chunks, attention_mask_chunks, is_query=False):
            vector_list, rnds = [], []
            self.model.requires_grad_(False)
            for input_ids, attention_mask in zip(inputs_ids_chunks, attention_mask_chunks):
                rnds.append(RandContext(input_ids, attention_mask))
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        inputs = {
                            "query": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if is_query else None,
                            "passage": {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                            } if not is_query else None,
                        }
                        outputs = self.model(
                            **inputs, only_encoding=True, return_dict=True
                        )
                        vector = outputs.query_vector if is_query else outputs.passage_vector
                vector_list.append(vector)
            self.model.requires_grad_(True)
            vector = torch.cat(vector_list, dim=0)
            return vector, rnds

        query_inputs_ids, query_attention_mask = query['input_ids'], query['attention_mask']
        query_inputs_ids_chunks = torch.chunk(query_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        query_attention_mask_chunks = torch.chunk(query_attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        query_vector, query_rnds = compute_vector(query_inputs_ids_chunks, query_attention_mask_chunks, is_query=True)

        passage_inputs_ids, passage_attention_mask = passage['input_ids'], passage['attention_mask']
        passage_inputs_ids_chunks = torch.chunk(passage_inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_attention_mask_chunks = torch.chunk(passage_attention_mask, chunks=self.training_args.gc_chunk_size,
                                                    dim=0)
        passage_vector, passage_rnds = compute_vector(passage_inputs_ids_chunks, passage_attention_mask_chunks)

        query_vector = query_vector.float().detach().requires_grad_()
        passage_vector = passage_vector.float().detach().requires_grad_()
        if self.training_args.negatives_x_device:
            all_query_vector = self.dist_gather_tensor(query_vector)
            all_passage_vector = self.dist_gather_tensor(passage_vector)
            decoder_scores = self.dist_gather_tensor(decoder_scores)
            retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
        else:
            retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        assert decoder_scores.size()[0] == retriever_scores.size()[0], (decoder_scores.size(), retriever_scores.size())
        teacher_scores = torch.zeros_like(retriever_scores, device=self.training_args.device)
        bsz = decoder_scores.size()[0]
        for i in range(bsz):
            start = i * self.data_args.train_n_passages
            end = (i + 1) * self.data_args.train_n_passages
            teacher_scores[i, start:end] = decoder_scores[i]

        retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
        retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores, reduction='batchmean')
        retriever_loss.backward()

        def vector_forward(inputs_ids_chunks, attention_mask_chunks, grads_chunk, rnds, is_query=False):
            for idx, (input_ids, attention_mask, grads, rnd) in \
                    enumerate(zip(inputs_ids_chunks, attention_mask_chunks, grads_chunk, rnds)):
                def chunk_forward():
                    with rnd:
                        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                            inputs = {
                                "query": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if is_query else None,
                                "passage": {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                } if not is_query else None,
                            }
                            outputs = self.model(
                                **inputs, only_encoding=True, return_dict=True
                            )
                            vector = outputs.query_vector if is_query else outputs.passage_vector
                            surrogate = torch.dot(vector.flatten().float(), grads.flatten())
                            sum_of_params = sum([p.sum() for p in self.model.parameters()])
                    surrogate = surrogate * self._dist_loss_scale_factor + 0. * sum_of_params
                    if self.use_amp:
                        self.scaler.scale(surrogate).backward()
                    else:
                        surrogate.backward()

                if idx != len(inputs_ids_chunks) - 1:
                    with self.model.no_sync():
                        chunk_forward()
                else:
                    chunk_forward()

        query_grads_chunk = torch.chunk(query_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        vector_forward(query_inputs_ids_chunks, query_attention_mask_chunks, query_grads_chunk, query_rnds,
                       is_query=True)

        passage_grads_chunk = torch.chunk(passage_vector.grad, chunks=self.training_args.gc_chunk_size, dim=0)
        vector_forward(passage_inputs_ids_chunks, passage_attention_mask_chunks, passage_grads_chunk, passage_rnds)

        return reader_loss + retriever_loss

    def separate_joint_encoding_compute_loss(self, inputs, global_step=None):

        _, reader_inputs, _, batch_pids = inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        if isinstance(batch_pids, tuple):
            batch_bias = batch_pids[1]
        else:
            batch_bias = torch.zeros(inputs_ids.size(0), device=inputs_ids.device)
        bias_chunks = torch.chunk(batch_bias, chunks=self.training_args.gc_chunk_size, dim=0)

        if global_step < self.training_args.distillation_start_steps or self.training_args.only_reader:
            reader_loss = 0
            for idx, (input_ids, attention_mask, independent_mask, labels, bias_scores) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks, bias_chunks)):
                def chunk_forward():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            independent_mask=independent_mask,
                            labels=labels,
                            use_cache=False,
                            #add_bias=self.training_args.add_bias,
                            #bias_scores=bias_scores,
                        ).loss
                        loss /= self.training_args.gc_chunk_size
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    return loss

                if idx != len(labels_chunks) - 1:
                    with self.model.no_sync():
                        loss = chunk_forward()
                else:
                    loss = chunk_forward()
                reader_loss = reader_loss + loss
            return reader_loss

        query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)

        def compute_colbert_scores(query_vector, passage_vector, query_mask, passage_mask):
            # [num_query, num_passages, q_len, p_len]
            score_list = []
            chunk_query_vector = torch.chunk(query_vector, chunks=query_vector.size()[0], dim=0)
            for chunk in chunk_query_vector:
                scores = chunk.unsqueeze(1) @ passage_vector.unsqueeze(0).transpose(2, 3)
                scores = scores.masked_fill(~passage_mask[None, :, None].bool(), -1e9)
                scores = torch.max(scores, dim=-1).values
                score_list.append(scores)
            scores = torch.cat(score_list, dim=0)
            scores = scores.masked_fill(~query_mask[:, None].bool(), 0.0)
            scores = torch.sum(scores, dim=-1) / query_mask.sum(dim=1)[..., None]

            return scores

        reader_loss = 0
        for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels, bias_scores) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks, bias_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        independent_mask=independent_mask,
                        query_mask=query_mask,
                        passage_mask=passage_mask,
                        labels=labels,
                        output_attentions=True,
                        return_dict=True,
                        use_cache=False,
                        #add_bias=self.training_args.add_bias,
                        #bias_scores=bias_scores,
                    )
                    loss = outputs.loss / self.training_args.gc_chunk_size
                    query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                    if len(query_vector.size()) == 3:
                        num_query, seq_len, _ = query_vector.size()
                        bsz = num_query // self.data_args.train_n_passages
                        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, seq_len, -1).mean(1)
                    else:
                        bsz = query_vector.size()[0] // self.data_args.train_n_passages
                        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    teacher_scores = scores.sum(dim=-1).mean(dim=1).detach()

                    if len(query_vector.size()) == 3:
                        retriever_scores = compute_colbert_scores(query_vector, passage_vector, query_mask,
                                                                  passage_mask)
                    else:
                        retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                        # retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1)) / \
                        #                    math.sqrt(query_vector.size()[-1])
                    retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
                    retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores,
                                                                reduction='batchmean') * self.training_args.retriever_weight
                    loss += retriever_loss / self.training_args.gc_chunk_size

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                return loss

            if idx != len(labels_chunks) - 1:
                with self.model.no_sync():
                    loss = chunk_forward()
            else:
                loss = chunk_forward()

            reader_loss = reader_loss + loss

        return reader_loss

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def train_step(self, batch, global_step=None):
        if self.training_args.separate_joint_encoding:
            return self.separate_joint_encoding_compute_loss(batch, global_step)

        if self.training_args.grad_cache:
            return self.grad_cache_compute_loss(batch, global_step)

        if self.use_amp:
            if version.parse(torch.__version__) > version.parse("1.7.1"):
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    kl_loss = self.compute_loss(batch, global_step)
            else:
                with torch.cuda.amp.autocast():
                    kl_loss = self.compute_loss(batch, global_step)
            self.scaler.scale(kl_loss).backward()
        else:
            kl_loss = self.compute_loss(batch, global_step)
            kl_loss.backward()

        return kl_loss

    def train(self):
        global_step = 0
        prev_global_step = 0
        prev_save_global_step = 0
        best_eval = 0.0
        if self.training_args.eval_at_start:
            eval_result = self.separate_joint_refresh_passages(do_eval=True, global_step=global_step)
            if isinstance(eval_result, tuple):
                if self.data_args.task == "XOR-Retrieve":
                    if self.training_args.only_reader:
                        eval_result = eval_result[1]
                    else:
                        eval_result = eval_result[0]
                elif self.data_args.task == "MIRACL":
                    eval_result = eval_result[0]
                else:
                    eval_result = eval_result[1]
            best_eval = eval_result
        if self.training_args.refresh_passages:
            if self.training_args.separate_joint_encoding:
                self.separate_joint_refresh_passages(do_eval=False)
            else:
                self.refresh_passages(do_eval=True, epoch=0)
        if self.training_args.max_steps > 0:
            dataset_epochs = [0] * len(self.train_dataloader)
        for epoch in range(self.num_train_epochs):
            if self.training_args.multi_task:
                for dataloader in self.train_dataloader:
                    if isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
            else:
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)
            self.epoch = copy.deepcopy(epoch)
            self.model.train()
            losses = AverageMeter('Loss', ':.4')
            progress = ProgressMeter(
                self.max_step if self.training_args.max_steps > 0 else self.num_training_steps,
                [losses],
                prefix="Epoch: [{}]".format(epoch))
            step = 0

            num_training_steps = [len(dataloader) for dataloader in self.train_dataloader]
            data_src_indices = []
            iterators = []
            for source, src_its in enumerate(num_training_steps):
                if self.training_args.max_steps > 0:
                    src_its = self.training_args.max_steps * self.training_args.gradient_accumulation_steps
                data_src_indices.extend([source] * src_its)
                train_dataloader = self.train_dataloader[source]
                iterators.append(iter(train_dataloader))

            epoch_rnd = random.Random(self.training_args.seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

            for i, source_idx in enumerate(data_src_indices):
                try:
                    it = iterators[source_idx]
                    batch = next(it)
                except:
                    if self.training_args.max_steps > 0:
                        dataset_epochs[source_idx] += 1
                        dataloader = self.train_dataloader[source_idx]
                        if isinstance(dataloader.sampler, DistributedSampler):
                            dataloader.sampler.set_epoch(dataset_epochs[source_idx])
                    iterators[source_idx] = iter(self.train_dataloader[source_idx])
                    it = iterators[source_idx]
                    batch = next(it)
                batch = self._prepare_inputs(batch)
                kl_loss = self.train_step(batch, global_step)

                if self.is_world_process_zero() and not self.training_args.only_reader:
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(name)

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                if self.training_args.negatives_x_device:
                    loss_list = [torch.zeros_like(kl_loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=loss_list, tensor=kl_loss.contiguous())
                    loss = torch.mean(torch.stack(loss_list, dim=0), dim=0)
                    losses.update(loss.item())
                else:
                    losses.update(kl_loss.item())

                step += 1
                if global_step != 0 and step % self.training_args.gradient_accumulation_steps == 0:
                    self._log_tensorboard_train(global_step, losses.val, losses.avg)
                    self._log_tensorboard_generations(global_step)
                if self.training_args.max_steps > 0 and self.training_args.gradient_accumulation_steps > 1:
                    if global_step != 0 and global_step != prev_global_step \
                            and global_step % self.training_args.print_steps == 0 and \
                            self.training_args.process_index in [-1, 0]:
                        progress.display(global_step)
                        prev_global_step = global_step
                else:
                    if step % (self.training_args.print_steps * self.training_args.gradient_accumulation_steps) == 0 \
                            and self.training_args.process_index in [-1, 0]:
                        progress.display(step // self.training_args.gradient_accumulation_steps)

                if global_step != 0 and global_step != prev_save_global_step and \
                        global_step % self.training_args.save_steps == 0 and \
                        global_step > self.training_args.distillation_start_steps:
                    prev_save_global_step = global_step

                    if self.training_args.only_reader:
                        scores = self._reader_validation_sample_scores(global_step)
                        if scores is None:
                            logger.warning("Skipping checkpoint-best: no validation sample is configured.")
                            eval_result = best_eval
                        else:
                            eval_result = scores["f1"]
                            logger.info("Reader validation sample at step %s: F1 %.4f, EM %.4f",
                                        global_step, scores["f1"], scores["em"])
                    elif self.training_args.separate_joint_encoding:
                        eval_result = self.separate_joint_refresh_passages(do_eval=True, global_step=global_step)
                    else:
                        eval_result = self.refresh_passages(epoch=self.epoch + 1)
                    if isinstance(eval_result, tuple):
                        if self.data_args.task == "XOR-Retrieve":
                            if self.training_args.only_reader:
                                eval_result = eval_result[1]
                            else:
                                eval_result = eval_result[0]
                        elif self.data_args.task == "MIRACL":
                            eval_result = eval_result[0]
                        else:
                            eval_result = eval_result[1]
                    if self.is_world_process_zero() and eval_result > best_eval:
                        best_eval = eval_result
                        checkpoint_folder = f"checkpoint-best"
                        output_dir = os.path.join(self.training_args.output_dir, checkpoint_folder)
                        self._save(self.model, output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        pids_path = os.path.join(self.training_args.output_dir, "dev_xor_retrieve_pids.jsonl")
                        if os.path.exists(pids_path):
                            shutil.copy2(pids_path, output_dir)
                        predictions_path = os.path.join(
                            self.training_args.output_dir, "dev_reader_xor_eng_span_predictions.json")
                        if os.path.exists(predictions_path):
                            shutil.copy2(predictions_path, output_dir)
                    if self.data_args.load_partial:
                        idx = global_step // self.training_args.save_steps
                        num_examples = 800 * self.training_args.save_steps
                        data_dir = self.data_args.train_dir
                        train_path = os.path.join(data_dir, self.data_args.train_path)
                        with open(train_path) as f:
                            examples = [jsonline for jsonline in f.readlines()[idx * num_examples: (idx + 1) * num_examples]]
                        self.train_dataloader[0].dataset.examples = examples

                if self.training_args.refresh_passages and global_step != 0 and global_step != self.max_step and \
                        global_step % self.training_args.refresh_intervals == 0:
                    if self.training_args.separate_joint_encoding:
                        self.separate_joint_refresh_passages(do_eval=False, global_step=global_step)
                    else:
                        self.refresh_passages(epoch=self.epoch + 1)

                if global_step >= self.max_step:
                    break
            if global_step >= self.max_step:
                break

    def test(self):
        if dist.is_available() and dist.is_initialized():
            torch.distributed.barrier()
        global_step = self.training_args.max_steps
        if self.training_args.eval_on_test:

            logger.info('Evaluating on MKQA')
            self.model = RRForConditionalGeneration.from_pretrained(
                os.path.join(self.training_args.output_dir, f"checkpoint-best"),
                config=self.config,
            )
            self.model.to(self.training_args.device)
            if self.data_args.task == "XOR-Retrieve":
                self.data_args.eval_query_file = "mkqa_dev_retrieve_eng_span.jsonl"
            elif self.data_args.task == "XOR-Full":
                self.data_args.eval_query_file = "mkqa_dev_full.jsonl"
            else:
                raise NotImplementedError
            _ = self.separate_joint_refresh_passages(do_eval=True, eval_set="mkqa", global_step=global_step)

    def refresh_passages(self, do_eval=True, epoch=0):
        self.model.eval()

        torch.cuda.empty_cache()
        output_dir = os.path.join(self.training_args.output_dir, "encoding")
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            self.encode(
                is_query=True, query_file=self.data_args.query_file,
                encoded_save_path=os.path.join(output_dir, "train_query_embedding.pt"),
                text_max_length=self.data_args.max_query_length,
            )
        torch.distributed.barrier()

        output_path = os.path.join(output_dir, "embedding_split%.2d.pt" % self.training_args.process_index)
        self.encode(is_query=False, encoded_save_path=output_path, text_max_length=self.data_args.max_passage_length)
        torch.distributed.barrier()
        logger.info(f"Process {self.training_args.process_index} Done encoding")

        eval_result = 0.0
        if self.is_world_process_zero():
            index_files = glob.glob(os.path.join(output_dir, "embedding_split*.pt"))

            p_reps_0, p_lookup_0 = torch.load(index_files[0])
            retriever = FaissIPRetriever(p_reps_0.float().numpy(), True)

            shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []

            all_p_reps = []
            for p_reps, p_lookup in shards:
                all_p_reps.append(p_reps.numpy())
                look_up += p_lookup
            retriever.add(np.concatenate(all_p_reps, axis=0))

            # load train query embeddings and refresh top-100 passages
            q_reps, q_lookup = torch.load(os.path.join(output_dir, "train_query_embedding.pt"))
            q_reps = q_reps.float().numpy()

            def search_queries(retriever, q_reps, p_lookup):
                all_scores, all_indices = retriever.batch_search(q_reps, 100, 5000)

                psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
                return all_scores, psg_indices

            all_scores, psg_indices = search_queries(retriever, q_reps, look_up)

            with open(os.path.join(output_dir, "train.jsonl"), 'w') as f:
                for qid, q_doc_scores, q_doc_indices in zip(q_lookup, all_scores, psg_indices):
                    score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
                    score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
                    example = {
                        'qid': qid,
                        'pids': [idx for _, idx in score_list],
                    }
                    f.write(json.dumps(example) + '\n')

            if do_eval:
                # load dev query embeddings and do evaluation
                self.encode(is_query=True, query_file="xor_dev_retrieve_eng_span_v1_1.jsonl",
                            encoded_save_path=os.path.join(output_dir, "dev_query_embedding.pt"),
                            text_max_length=self.data_args.max_query_length)
                q_reps, q_lookup = torch.load(os.path.join(output_dir, "dev_query_embedding.pt"))
                q_reps = q_reps.float().numpy()

                all_scores, psg_indices = search_queries(retriever, q_reps, look_up)
                write_ranking(psg_indices, all_scores, q_lookup,
                              os.path.join(output_dir, "dev_xor_retrieve_results.jsonl"),
                              os.path.join(self.data_args.train_dir, "xor_dev_retrieve_eng_span_v1_1.jsonl"),
                              self.train_dataset[0].corpus, False)

                predictions = json.load(open(os.path.join(output_dir, "dev_xor_retrieve_results.jsonl")))
                input_data = read_jsonlines(
                    os.path.join(self.data_args.train_dir, "xor_dev_retrieve_eng_span_v1_1.jsonl"))
                qid2answers = {item["id"]: item["answers"] for item in input_data}
                eval_results = {}
                for topk in [2, 5]:
                    logger.info("Evaluating R@{}kt".format(topk))
                    pred_per_lang_results = evaluate_top_k_hit(
                        predictions, qid2answers, topk * 1000)
                    avg_scores = []
                    for lang in pred_per_lang_results:
                        logger.info(
                            "performance on {0} ({1} examples)".format(lang, pred_per_lang_results[lang]["count"]))
                        per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang][
                            "count"]) * 100
                        logger.info(per_lang_score)

                        avg_scores.append(per_lang_score)

                    logger.info("Final macro averaged score: ")
                    logger.info(mean(avg_scores))
                    eval_results[topk] = mean(avg_scores)
                eval_result = eval_results[2]

        torch.distributed.barrier()
        with open(os.path.join(output_dir, "train.jsonl"), 'r') as f:
            examples = [json.loads(jsonline) for jsonline in f]
        self.train_dataloader[0].dataset.examples = examples
        self.model.train()

        return eval_result

    def encode(self, is_query=False, query_file=None, encoded_save_path=None, text_max_length=200):
        if is_query:
            queries = GenericDataLoader(self.data_args.train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=query_file).load_queries()
            start, end = 0, len(queries)
        else:
            corpus = self.train_dataset[0].corpus
            shard_size = len(corpus) // self.world_size
            start = self.training_args.process_index * shard_size
            end = (self.training_args.process_index + 1) * shard_size \
                if self.training_args.process_index + 1 != self.world_size else len(corpus)
            logger.info(
                f'Process {self.training_args.process_index} => Generate passage embeddings from {start} to {end}')

        encode_dataset = EncodeDataset(queries if is_query else corpus, self.tokenizer,
                                       max_length=text_max_length, is_query=is_query,
                                       start=start, end=end, normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case,
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=True, )
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=text_max_length
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        encoded = []
        lookup_indices = []

        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    if self.data_args.encode_is_qry:
                        query_vector = self.model(query=batch.data, only_encoding=True).query_vector
                        encoded.append(query_vector.cpu())
                    else:
                        passage_vector = self.model(passage=batch.data, only_encoding=True).passage_vector
                        encoded.append(passage_vector.cpu())

        encoded = torch.cat(encoded)
        torch.save((encoded, lookup_indices), encoded_save_path)

    def separate_joint_refresh_passages(self, do_eval=True, eval_set="dev", global_step=0):
        if not do_eval and self.training_args.use_mcontriever and global_step < self.training_args.self_retrieve_steps:
            logger.info(f"Process {self.training_args.process_index} Loading updated training passages")
            with open(os.path.join(self.data_args.train_dir, "mss.ICL.train.jsonl"), 'r') as f:
                examples = [json.loads(jsonline) for jsonline in f]
            idx = global_step // self.training_args.refresh_intervals
            num_queries = 64 * self.training_args.refresh_intervals
            examples = examples[idx * num_queries: (idx + 1) * num_queries]
            self.train_dataloader[0].dataset.examples = examples

            return

        self.model.eval()

        torch.cuda.empty_cache()

        if ('nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir) and do_eval \
                and self.data_args.task != "MIRACL":
            train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" \
                else 'data/XOR-Full'
            corpus_file = 'psgs_w100.tsv' if self.data_args.task == "XOR-Retrieve" else 'all_w100.tsv'
            corpus = GenericDataLoader(train_dir, corpus_file=corpus_file,
                                       query_file=self.data_args.query_file).load_corpus()
        else:
            corpus = self.train_dataset[0].corpus
            train_dir = self.data_args.train_dir

        results = self.separate_joint_encode(do_eval=do_eval, corpus=corpus, global_step=global_step)
        logger.info(f"Process {self.training_args.process_index} Done encoding")

        def save_results(results, output_path, add_score=True):
            with open(output_path, 'w') as f:
                for qid in results:
                    sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                    example = {
                        'qid': qid,
                        'pids': [(docid, score) if add_score else docid for docid, score in sorted_indices_scores]
                    }
                    f.write(json.dumps(example) + '\n')

        refresh = not (global_step == 0 and os.path.exists(os.path.join(self.training_args.output_dir, "train.jsonl")))

        if do_eval:
            output_path = os.path.join(self.training_args.output_dir, "{}.split{}.jsonl".format(
                eval_set, self.training_args.process_index))
            save_results(results, output_path)
        else:
            if refresh:
                output_path = os.path.join(self.training_args.output_dir,
                                           "train.split{}.jsonl".format(self.training_args.process_index))
                save_results(results, output_path)

        torch.distributed.barrier()

        eval_result = 0.0
        if self.is_world_process_zero():
            def load_results(data_path):
                prediction_files = sorted(glob.glob(data_path))
                results = {}
                for path in prediction_files:
                    with open(path) as f:
                        for jsonline in f.readlines():
                            example = json.loads(jsonline)
                            qid = example['qid']
                            if qid not in results:
                                results[qid] = {}
                            for pid, score in example['pids']:
                                if not do_eval and 'mss' in self.data_args.query_file and len(qid.split("-")) >= 2 \
                                        and pid == qid.split("-")[1]:
                                    continue
                                results[qid][pid] = score
                return results

            if do_eval:
                results = load_results(os.path.join(self.training_args.output_dir, f"{eval_set}.split*.jsonl"))

                if self.data_args.task == "MIRACL":
                    output_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_pids.jsonl")
                    with open(output_path, 'w') as f:
                        for qid in results:
                            sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                            for k, (docid, score) in enumerate(sorted_indices_scores):
                                f.write(f'{qid} Q0 {docid} {k + 1} {score} dense\n')

                    import subprocess
                    file_path = '-m pyserini.eval.trec_eval'
                    args = f'-c -m ndcg_cut.10 -m recall.100 ' \
                           f'{os.path.join(train_dir, self.data_args.eval_query_file.replace("topics", "qrels"))} ' \
                           f'{output_path}'
                    result = subprocess.run(['python', *file_path.split(), *args.split()], capture_output=True, text=True)
                    output = result.stdout
                    logger.info(output)
                    eval_result = re.findall(r'\d+\.\d+', output)[-1]
                    eval_result = (float(eval_result),)
                else:
                    from retriever import parse_qa_jsonlines_file
                    qas_file = os.path.join(train_dir, self.data_args.eval_query_file)
                    qas = {}
                    for question, qid, answers, lang in parse_qa_jsonlines_file(qas_file):
                        qas[qid] = (question, answers, lang)

                    xor_output_prediction_format = []
                    reader_evaluation_format = []
                    output_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_results.json")
                    with open(output_path, 'w') as f:
                        for qid in results:
                            sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                            ctxs, pids = [], []
                            for docid, score in sorted_indices_scores:
                                ctxs.append(corpus[docid]["text"])
                                pids.append(docid)
                            question, answers, lang = qas[qid]
                            xor_output_prediction_format.append({"id": qid, "lang": lang, "ctxs": ctxs})
                            reader_evaluation_format.append({"qid": qid, "pids": pids})
                        json.dump(xor_output_prediction_format, f)
                    output_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_pids.jsonl")
                    with open(output_path, 'w') as f:
                        for example in reader_evaluation_format:
                            f.write(json.dumps(example) + '\n')

                    if eval_set != "test":
                        predictions = json.load(open(os.path.join(self.training_args.output_dir,
                                                                  "dev_xor_retrieve_results.json")))
                        input_data = read_jsonlines(
                            os.path.join(train_dir, self.data_args.eval_query_file))
                        qid2answers = {item["id"]: item["answers"] for item in input_data}
                        eval_results = {}
                        for topk in [2, 5, 100]:
                            logger.info("Evaluating R@{}kt".format(topk))
                            pred_per_lang_results = evaluate_top_k_hit(
                                predictions, qid2answers, topk * 1000)
                            avg_scores = []
                            for lang in pred_per_lang_results:
                                logger.info(
                                    "performance on {0} ({1} examples)".format(lang,
                                                                               pred_per_lang_results[lang]["count"]))
                                per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang][
                                    "count"]) * 100
                                logger.info(per_lang_score)

                                avg_scores.append(per_lang_score)

                            logger.info("Final macro averaged score: ")
                            logger.info(mean(avg_scores))
                            eval_results[topk] = mean(avg_scores)
                        eval_result = eval_results[2]
            else:
                if refresh:
                    results = load_results(os.path.join(self.training_args.output_dir, "train.split*.jsonl"))
                    output_path = os.path.join(self.training_args.output_dir, "train.jsonl")
                    save_results(results, output_path, add_score=False)

        torch.distributed.barrier()
        if not do_eval:
            logger.info(f"Process {self.training_args.process_index} Loading updated training passages")
            with open(os.path.join(self.training_args.output_dir, "train.jsonl"), 'r') as f:
                examples = [json.loads(jsonline) for jsonline in f]
            self.train_dataloader[0].dataset.examples = examples
        elif self.data_args.task != "MIRACL":
            self.eval_reader(corpus=corpus, eval_set=eval_set)
            torch.distributed.barrier()

            if self.is_world_process_zero():
                data_path = os.path.join(self.training_args.output_dir,
                                         f"{eval_set}_reader_xor_eng_span_predictions.split*.json")
                prediction_files = sorted(glob.glob(data_path))
                results = {}
                for path in prediction_files:
                    with open(path) as f:
                        results.update(json.load(f))
                output_path = os.path.join(self.training_args.output_dir,
                                           f"{eval_set}_reader_xor_eng_span_predictions.json")
                with open(output_path, 'w') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                import subprocess

                if eval_set != "test":
                    if self.data_args.task == 'XOR-Retrieve':
                        file_path = 'evals/eval_xor_engspan.py'
                        args = f'--data_file {os.path.join(train_dir, self.data_args.eval_query_file)} ' \
                               f'--pred_file {output_path}'

                        result = subprocess.run(['python', file_path, *args.split()], capture_output=True, text=True)
                        output = result.stdout
                        pattern = re.compile(r'F1: (\d+\.\d+), EM:(\d+\.\d+)')
                        matches = pattern.findall(output)
                        f1_score = [float(match[0]) for match in matches][-1]
                        em_score = [float(match[1]) for match in matches][-1]
                        logger.info(output)
                        eval_result = (eval_result, f1_score, em_score)
                    else:
                        assert self.data_args.task == "XOR-Full", self.data_args.task
                        file_path = 'evals/eval_xor_full.py'
                        args = f'--data_file data/XOR-Full/{self.data_args.eval_query_file} ' \
                               f'--pred_file {output_path}'
                        result = subprocess.run(['python', file_path, *args.split()], capture_output=True, text=True)
                        output = result.stdout
                        pattern = re.compile(r'avg f1: (\d+\.\d+)\navg em: (\d+\.\d+)\navg bleu: (\d+\.\d+)')
                        matches = pattern.findall(output)
                        f1_score = [float(match[0]) for match in matches][-1]
                        em_score = [float(match[1]) for match in matches][-1]
                        bleu_score = [float(match[2]) for match in matches][-1]
                        logger.info(output)
                        eval_result = (eval_result, f1_score, em_score, bleu_score)
            torch.distributed.barrier()

        self.model.train()

        return eval_result

    def eval_reader(self, corpus, eval_set="dev"):
        if 'nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir:
            train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" \
                else 'data/XOR-Full'
        else:
            train_dir = self.data_args.train_dir
        queries = GenericDataLoader(train_dir, corpus_file=self.data_args.corpus_file,
                                    query_file=self.data_args.eval_query_file).load_queries()

        train_path = os.path.join(self.training_args.output_dir, f"{eval_set}_xor_retrieve_pids.jsonl")
        with open(train_path) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        shard_size = len(examples) // self.training_args.world_size
        start = self.training_args.process_index * shard_size
        end = (self.training_args.process_index + 1) * shard_size \
            if self.training_args.process_index + 1 != self.training_args.world_size else len(examples)
        examples = examples[start:end]

        eval_dataset = ReaderDataset(
            queries=queries,
            corpus=corpus,
            tokenizer=self.tokenizer,
            train_path=examples,
            data_args=self.data_args,
            eval_mode=True,
        )

        data_collator = ReaderCollator(
            self.tokenizer,
            max_query_length=self.data_args.max_query_length,
            max_passage_length=self.data_args.max_passage_length,
            max_query_passage_length=self.data_args.max_query_passage_length,
            max_answer_length=self.data_args.max_answer_length,
            separate_joint_encoding=self.training_args.separate_joint_encoding,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=2,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        model = self.model
        while hasattr(model, 'module'):
            model = model.module

        train_n_passages = 100 if self.data_args.train_n_passages == 1 else self.data_args.train_n_passages
        model.n_passages = train_n_passages
        predictions = {}
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                qids, reader_inputs, batch_answer, batch_pids = batch

                bsz, seq_len = reader_inputs["input_ids"].size()
                input_ids = reader_inputs["input_ids"].view(bsz // train_n_passages, train_n_passages, seq_len)
                model_inputs = {
                    "input_ids": input_ids.to(self.training_args.device),
                    "attention_mask": reader_inputs["attention_mask"].to(self.training_args.device),
                    "independent_mask": reader_inputs["independent_mask"].to(self.training_args.device),
                    "add_bias": self.training_args.add_bias,
                }

                outputs = model.generate(
                    **model_inputs,
                    max_length=self.data_args.max_answer_length,
                    num_beams=1,
                )
                for k, o in enumerate(outputs):
                    ans = self.tokenizer.decode(o, skip_special_tokens=True)
                    predictions[qids[k]] = ans

                torch.cuda.empty_cache()

        output_path = os.path.join(self.training_args.output_dir, "{}_reader_xor_eng_span_predictions.split{}.json".
                                   format(eval_set, self.training_args.process_index))
        with open(output_path, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        model.n_passages = self.data_args.train_n_passages

    def encode_query(self, queries, start, end, do_eval=False):
        encode_dataset = EncodeDataset(queries, self.tokenizer, max_length=self.data_args.max_query_length,
                                       is_query=True, start=start, end=end,
                                       normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case, separate_joint_encoding=True,
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=do_eval,
                                       task=self.data_args.task)
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=self.data_args.max_query_length,
                separate_joint_encoding=True,
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )
        encoded = []
        mask = []
        lookup_indices = []

        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    query_vector = self.model(query=batch, only_encoding=True).query_vector
                    encoded.append(query_vector)
                    mask.append(batch['attention_mask'])

        encoded = torch.cat(encoded)
        mask = torch.cat(mask)
        return encoded, mask, lookup_indices

    def separate_joint_encode(self, corpus, do_eval=True, global_step=0):
        if self.data_args.query_file != "fs-qa.cl+il.llm-qa.jsonl" and global_step == 0 and \
                os.path.exists(os.path.join(self.training_args.output_dir, "train.jsonl")):
            if self.data_args.query_file == "xor_nq_train_full.jsonl":
                queries = GenericDataLoader(self.training_args.output_dir, corpus_file=self.data_args.corpus_file,
                                            query_file="train.queries.jsonl").load_queries()
                assert len(queries) == 104486, len(queries)
                self.train_dataloader[0].dataset.queries = queries
            return

        if do_eval:
            if ('nq-dpr' in self.data_args.train_dir or self.data_args.task not in self.data_args.train_dir) \
                    and self.data_args.task != "MIRACL":
                train_dir = 'data/XOR-Retrieve' if self.data_args.task == "XOR-Retrieve" \
                    else 'data/XOR-Full'
            else:
                train_dir = self.data_args.train_dir
            queries = GenericDataLoader(train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.eval_query_file).load_queries()

            if self.data_args.query_file == "xor_train_full_il.jsonl":
                new_queries = {}
                for qid in queries.keys():
                    query, answers, langs = queries[qid]
                    new_queries[qid] = (query[5:].strip(), answers, langs)
                queries = new_queries
        else:
            queries = GenericDataLoader(self.data_args.train_dir, corpus_file=self.data_args.corpus_file,
                                        query_file=self.data_args.query_file).load_queries()
            if ('mss' in self.data_args.query_file or 'wikidata' in self.data_args.query_file
                    or 'fs-qa' in self.data_args.query_file) and len(queries) >= 200000:
                idx = global_step // self.training_args.refresh_intervals
                num_queries = self.training_args.train_batch_size * self.training_args.world_size * self.training_args.refresh_intervals
                start = idx * num_queries if idx * num_queries < len(queries) else idx * num_queries - len(queries)
                end = start + num_queries
                shard_queries = dict(list(queries.items())[start: end])
                if len(shard_queries) < num_queries:
                    offset = num_queries - len(shard_queries)
                    shard_queries.update(dict(list(queries.items())[:offset]))
                queries = shard_queries
                assert len(queries) == num_queries, (len(queries), num_queries)

            if self.data_args.query_file == "xor_nq_train_full.jsonl":
                if self.is_world_process_zero():
                    new_queries = {}
                    for qid in queries.keys():
                        query, answers, langs = queries[qid]
                        if 'nq' in qid:
                            idx = random.choice(range(len(answers)))
                            answer = answers[idx]
                            lang = langs[idx]
                            new_queries[qid] = (f"[{lang}] " + query, [answer], lang)
                        else:
                            new_queries[qid] = (query, answers, langs)
                    with open(os.path.join(self.training_args.output_dir, "train.queries.jsonl"), 'w') as f:
                        for qid, (query, answer, lang) in new_queries.items():
                            f.write(json.dumps({"id": qid, "question": query, "answers": answer, "lang": lang}) + '\n')
                torch.distributed.barrier()
                queries = GenericDataLoader(self.training_args.output_dir, corpus_file=self.data_args.corpus_file,
                                            query_file="train.queries.jsonl").load_queries()
                assert len(queries) == 104486, len(queries)
                self.train_dataloader[0].dataset.queries = queries

        if self.training_args.debug:
            queries = dict(list(queries.items())[:100])
        start, end = 0, len(queries)
        query_vector, q_mask, q_lookup_indices = self.encode_query(queries, start, end, do_eval=do_eval)

        results = {qid: {} for qid in q_lookup_indices}

        if self.training_args.debug:
            corpus = dict(list(corpus.items())[:1000])

        shard_size = len(corpus) // self.training_args.world_size
        start = self.training_args.process_index * shard_size
        end = (self.training_args.process_index + 1) * shard_size \
            if self.training_args.process_index + 1 != self.training_args.world_size else len(corpus)
        logger.info(
            f'Process {self.training_args.process_index} => Generate passage embeddings from {start} to {end}')

        encode_dataset = EncodeDataset(corpus, self.tokenizer, max_length=self.data_args.max_passage_length,
                                       is_query=False, start=start, end=end,
                                       normalize_text=self.data_args.normalize_text,
                                       lower_case=self.data_args.lower_case, separate_joint_encoding=True,
                                       add_lang_token=self.data_args.add_lang_token,
                                       eval_mode=do_eval,
                                       task=self.data_args.task)
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=self.training_args.per_device_eval_batch_size * self.training_args.n_gpu,
            collate_fn=EncodeCollator(
                self.tokenizer,
                max_length=self.data_args.max_passage_length,
                separate_joint_encoding=True,
                padding_to_max_length=True,
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

        lookup_indices, batch_scores = [], []
        for bidx, (batch_ids, batch) in enumerate(tqdm(encode_loader)):
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                lookup_indices.extend(batch_ids)
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.training_args.device)
                    # [bz, p_len, dim]
                    passage_vector = self.model(passage=batch, only_encoding=True).passage_vector
                    passage_mask = batch['attention_mask']

                    if len(query_vector.size()) == 3:
                        scores = compute_colbert_scores(query_vector, passage_vector, q_mask, passage_mask)
                    else:
                        scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                    batch_scores.append(scores)

            if len(batch_scores) % 100 == 0 or bidx == len(encode_loader) - 1:
                batch_scores = torch.cat(batch_scores, dim=1)
                if batch_scores.size()[0] > 64000:
                    batch_scores_list = torch.chunk(batch_scores, chunks=batch_scores.size()[0] // 32000, dim=0)
                    sorted_scores_list, sorted_indices_list = [], []
                    for batch_scores in batch_scores_list:
                        sorted_scores, sorted_indices = torch.topk(batch_scores, k=100, dim=-1)
                        sorted_scores_list.append(sorted_scores)
                        sorted_indices_list.append(sorted_indices)
                    sorted_scores = torch.cat(sorted_scores_list, dim=0)
                    sorted_indices = torch.cat(sorted_indices_list, dim=0)
                else:
                    sorted_scores, sorted_indices = torch.topk(batch_scores, k=min(100, batch_scores.size(-1)), dim=-1)
                sorted_scores = sorted_scores.cpu().numpy().tolist()
                sorted_indices = sorted_indices.cpu().numpy().tolist()
                for i, (scores, indices) in enumerate(zip(sorted_scores, sorted_indices)):
                    qid = q_lookup_indices[i]
                    for score, idx in zip(scores, indices):
                        docid = lookup_indices[idx]
                        results[qid][docid] = score
                    if len(results[qid]) > 100:
                        sorted_indices_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:100]
                        results[qid] = {docid: score for docid, score in sorted_indices_scores}
                lookup_indices, batch_scores = [], []

        return results


class ReaderWikidataTrainer(ReaderTrainer):
    def __init__(self, model, train_dataset, data_collator, training_args, data_args, tokenizer):
        super(ReaderWikidataTrainer, self).__init__(model, train_dataset, data_collator, training_args, data_args,
                                                    tokenizer)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def compute_loss(self, inputs, global_step=None):
        _, reader_inputs, _, _ = inputs

        input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            independent_mask=independent_mask,
            query_mask=query_mask,
            passage_mask=passage_mask,
            labels=labels,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        reader_loss = outputs.loss
        query_vector, passage_vector = outputs.query_vector, outputs.passage_vector

        bsz = query_vector.size()[0] // self.data_args.train_n_passages
        query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

        if self.training_args.negatives_x_device:
            all_query_vector = self.dist_gather_tensor(query_vector)
            all_passage_vector = self.dist_gather_tensor(passage_vector)
            retriever_scores = torch.matmul(all_query_vector, all_passage_vector.transpose(0, 1))
        else:
            retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

        target = torch.arange(
            retriever_scores.size(0),
            device=retriever_scores.device,
            dtype=torch.long
        )

        retriever_loss = self.cross_entropy(retriever_scores, target)

        if self.training_args.negatives_x_device:
            retriever_loss = retriever_loss * self.world_size
        retriever_loss = retriever_loss * self.training_args.retriever_weight

        loss = reader_loss + retriever_loss
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return reader_loss + retriever_loss / self._dist_loss_scale_factor

    def e2e_compute_loss(self, inputs, global_step=None):
        _, reader_inputs, _, _ = inputs

        inputs_ids, attention_mask, independent_mask, query_mask, passage_mask, labels = \
            reader_inputs['input_ids'], reader_inputs['attention_mask'], reader_inputs['independent_mask'], \
            reader_inputs['query_mask'], reader_inputs['passage_mask'], reader_inputs['labels']
        self.training_args.gc_chunk_size = labels.size()[0]
        input_ids_chunks = torch.chunk(inputs_ids, chunks=self.training_args.gc_chunk_size, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        independent_mask_chunks = torch.chunk(independent_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        labels_chunks = torch.chunk(labels, chunks=self.training_args.gc_chunk_size, dim=0)

        if global_step < self.training_args.distillation_start_steps or self.training_args.only_reader:
            reader_loss = 0
            for idx, (input_ids, attention_mask, independent_mask, labels) in \
                    enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, labels_chunks)):
                def chunk_forward():
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            independent_mask=independent_mask,
                            labels=labels,
                            use_cache=False,
                        ).loss
                        loss /= self.training_args.gc_chunk_size
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    return loss

                if idx != len(labels_chunks) - 1:
                    with self.model.no_sync():
                        loss = chunk_forward()
                else:
                    loss = chunk_forward()
                reader_loss = reader_loss + loss
            return reader_loss

        query_mask_chunks = torch.chunk(query_mask, chunks=self.training_args.gc_chunk_size, dim=0)
        passage_mask_chunks = torch.chunk(passage_mask, chunks=self.training_args.gc_chunk_size, dim=0)

        reader_loss = 0
        for idx, (input_ids, attention_mask, independent_mask, query_mask, passage_mask, labels) in \
                enumerate(zip(input_ids_chunks, attention_mask_chunks, independent_mask_chunks, query_mask_chunks,
                              passage_mask_chunks, labels_chunks)):
            def chunk_forward():
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        independent_mask=independent_mask,
                        query_mask=query_mask,
                        passage_mask=passage_mask,
                        labels=labels,
                        output_attentions=True,
                        return_dict=True,
                        use_cache=False,
                    )
                    loss = outputs.loss / self.training_args.gc_chunk_size
                    query_vector, passage_vector = outputs.query_vector, outputs.passage_vector
                    bsz = query_vector.size()[0] // self.data_args.train_n_passages
                    query_vector = query_vector.view(bsz, self.data_args.train_n_passages, -1).mean(1)

                    cross_attention_scores = outputs.cross_attentions[-1][:, :, 0]
                    bsz, n_heads, _ = cross_attention_scores.size()
                    scores = cross_attention_scores.view(bsz, n_heads, self.data_args.train_n_passages, -1)
                    teacher_scores = scores.sum(dim=-1).mean(dim=1).detach()

                    retriever_scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))
                    retriever_logits = torch.log_softmax(retriever_scores, dim=-1)
                    retriever_loss = torch.nn.functional.kl_div(retriever_logits, teacher_scores,
                                                                reduction='batchmean') * self.training_args.retriever_weight
                    loss += retriever_loss / self.training_args.gc_chunk_size

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                return loss

            if idx != len(labels_chunks) - 1:
                with self.model.no_sync():
                    loss = chunk_forward()
            else:
                loss = chunk_forward()

            reader_loss = reader_loss + loss

        return reader_loss

    def train_step(self, batch, global_step=None):
        if self.training_args.e2e_training:
            return self.e2e_compute_loss(batch, global_step)
        return self.compute_loss(batch, global_step)

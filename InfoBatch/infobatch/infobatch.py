import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings
import faiss
__all__ = ['InfoBatch']


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                "IterableDataset replica at each worker. Please see "
                                "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


class InfoBatch(Dataset):
    """
    InfoBatch aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875, nbits: int = 12, alpha: int = 8, hash_codes=None):
        self.dataset = dataset
        #保留比例
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        #生成一个包含所有元素值为 1 的张量，形状为 [样本数]。* 3：将所有元素的值乘以 3，即所有样本的初始分数设为 3。
        self.scores = torch.ones(len(self.dataset)) * 3# 初始化所有样本的分数
        self.weights = torch.ones(len(self.dataset))

        # 检查并保存传入的哈希桶号
        if hash_codes is not None:
            assert len(hash_codes) == len(dataset), "Hash codes length must match dataset length."
            print("Hash codes successfully initialized in InfoBatch.")
        else:
            print("Warning: No hash codes provided to InfoBatch.")
        # 保存传入的哈希桶号
        self.hash_codes = hash_codes if hash_codes is not None else [None] * len(dataset)

        # 初始化 IndexLSH
        self.d = dataset[0][0].shape[0]  # 获取样本维度
        # self.index_lsh = faiss.IndexLSH(self.d, nbits)
        # self.index_lsh = faiss.IndexLSH(8, 64)
        self.alpha = alpha  # 分段数

        self.num_pruned_samples = 0
        self.cur_batch_index = None

    # def __getattr__(self, name):
        # Delegate the method call to the self.dataset if it is not found in Wrapper
        # if name == "sampler":
            # raise AttributeError("Direct access to sampler is not allowed")
        # if name == "sampler":
            # return self.sampler
        # return getattr(self.dataset, name)
    def __getattr__(self, name):
        # Delegate the method call to the self.dataset if it is not found in Wrapper
        return getattr(self.dataset, name)

    # 用于更新 InfoBatch 类中的 cur_batch_index 属性,即当前 batch 的样本索引
    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, values):

        # print("Update values shape:", values.shape)  # 验证values的形状是否为逐样本损失

        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)

        # 打印调试信息
        # print(f"Updating scores with indices: {indices}")
        # print(f"Values (loss) received: {values}")

        loss_val = values.detach().clone()
        self.cur_batch_index = []

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1), loss_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            loss_val = iv_whole_group[1]
        self.scores[indices.cpu().long()] = loss_val.cpu()
        # print(f"Updated scores for indices {indices.cpu().long()}: {self.scores[indices.cpu().long()]}")

        values.mul_(weights)
        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index] # , index
        # return self.dataset[index], index, self.scores[index]



    def prune(self):
        # print("Pruning started...")
        well_learned_mask = (self.scores < self.scores.mean()).numpy()
        well_learned_indices = np.where(well_learned_mask)[0]
        # print("keep_ratio is like: ", self.keep_ratio)
        selected_indices_well = np.random.choice(well_learned_indices, int(
            self.keep_ratio * len(well_learned_indices)), replace=False)# 对低分样本进行剪枝后剩余的索引值
        # print("number of selected_indices_well is", len(selected_indices_well))
        self.reset_weights()
        if len(selected_indices_well) > 0:
            self.weights[selected_indices_well] = 1 / self.keep_ratio
        # remained_indices = np.where(~well_learned_mask)[0].tolist()# 高分样本列表
        high_score_indices = np.where(~well_learned_mask)[0]# 高分样本索引
        pruned_indices = selected_indices_well.tolist()
        # print("number of pruned_indices_1 is", len(pruned_indices))

        # 打印低分样本与高分样本数量
        # print(f"Low-score samples to prune: {len(well_learned_indices)}, High-score samples: {len(high_score_indices)}")

        # Step 1: 高分样本排序
        sorted_high_score_indices = high_score_indices[np.argsort(-self.scores[high_score_indices])]

        # Step 2: 等深分段
        alpha = 8  # 分段数量
        segments = np.array_split(sorted_high_score_indices, alpha)

        # Step 3: 使用哈希分桶
        buckets = []  # 用于存储每个段的桶

        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue
            # print(f"Segment {i + 1}/{self.alpha} with {len(segment)} samples.")
            buckets_for_segment = {}
            for idx in segment:
                # 使用哈希值作为桶ID
                bucket_id = self.hash_codes[idx]
                if bucket_id not in buckets_for_segment:
                    buckets_for_segment[bucket_id] = []
                buckets_for_segment[bucket_id].append(idx)
            # print(f"  Created {len(buckets_for_segment)} buckets in Segment {i + 1}")
            buckets.append(buckets_for_segment)
            # print("Step 3 Completed - Total Buckets Generated:", len(buckets))

            # Step 3.1：为每个段增加扰动项以生成多维输入
            # segment_scores = self.scores[segment].cpu().numpy().reshape(-1, 1).astype('float32')
            # noise = np.random.normal(0, 1, (len(segment), 4)).astype('float32')  # 添加4维随机扰动
            # segment_scores_with_noise = np.hstack((segment_scores, noise))  # 将扰动项合并到segment_scores
            # 创建IndexLSH并添加多维特征
            # index_lsh = faiss.IndexLSH(segment_scores_with_noise.shape[1], nbits)
            # index_lsh.train(segment_scores_with_noise)  # 确保LSH索引训练完成
            # 编码样本得到哈希值
            # hash_codes = np.zeros((len(segment), (nbits + 7) // 8), dtype=np.uint8)
            # index_lsh.sa_encode(segment_scores_with_noise, hash_codes)

            # Step 3.4：将哈希值转换为桶ID并分桶
            # buckets_for_segment = {}
            # for j, hash_code in enumerate(hash_codes):
                # bucket_id = hash_code.tobytes()  # 使用哈希字节表示作为桶ID
                # if bucket_id not in buckets_for_segment:
                    # buckets_for_segment[bucket_id] = []
                # buckets_for_segment[bucket_id].append(segment[j])

            # buckets.append(buckets_for_segment)

        # Step 4: 桶内逐桶剪枝
        # print("Starting Step 4 - Pruning within each bucket")
        # pruned_indices = []

        for group_idx, bucket_group in enumerate(buckets):
            # print(f"Processing Bucket Group {group_idx + 1}/{len(buckets)}")
            for bucket_id, indices in bucket_group.items():
                # print(f"  Bucket {bucket_id}: Sample Count = {len(indices)}")

                # 如果桶内有多个样本，保留最高分的样本
                if len(indices) > 1:
                    highest_score_index = indices[np.argmax(self.scores[indices])]
                    pruned_indices.append(highest_score_index)
                    # print(f"    Retained Highest Score Sample Index: {highest_score_index}")

                    # 剩余样本进行按比例剪枝
                    num_to_keep = max(1, int(self.keep_ratio * (len(indices) - 1)))
                    selected_indices = np.random.choice(
                        [idx for idx in indices if idx != highest_score_index],
                        num_to_keep,
                        replace=False
                    )
                    pruned_indices.extend(selected_indices)
                    # print(f"    Randomly Selected {num_to_keep} Samples for Retention: {selected_indices}")

                    # 按比例调整被选中样本的权重
                    self.weights[selected_indices] = 1 / self.keep_ratio
                else:
                    pruned_indices.extend(indices)
                    # print(f"    Single Sample in Bucket, Retained without Pruning: {indices[0]}")
        # pruned_indices.extend(selected_indices_well)
        # print("number of pruned_indices_2 is", len(pruned_indices))
        self.num_pruned_samples += len(self.dataset) - len(pruned_indices)
        np.random.shuffle(pruned_indices)

        return pruned_indices

    '''

        def prune(self):
        well_learned_mask = (self.scores < self.scores.mean()).numpy()
        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        high_score_indices = np.where(~well_learned_mask)[0]

        # Step 1: 高分样本排序
        sorted_high_score_indices = high_score_indices[np.argsort(-self.scores[high_score_indices])]
        # print("Step 1 - Sorted high-score indices:", sorted_high_score_indices)

        # Step 2: 等深分段
        alpha = 8  # 选择的分段数量
        segments = np.array_split(sorted_high_score_indices, alpha)
        # print("Step 2 - Segments:", segments)

        # Step 3: 使用哈希分桶
        # print("Starting Step 3 - Hash Bucketing")
        buckets = []  # 用于存储每个段的桶
        nbits = 64  # 哈希位数

        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            # Step 3.1：创建哈希输入向量
            # print(f"Creating hash input for segment {i}")
            # 这里将 `self.scores[segment]` 转换为 `numpy` 数组以使用 `astype`
            print(f"\nSegment {i}: Number of Samples = {len(segment)}")
            segment_scores = self.scores[segment].cpu().numpy().reshape(-1, 1).astype('float32')
            print(f"  Hash Input Vector - dtype: {segment_scores.dtype}, shape: {segment_scores.shape}")
            # print(f"segment_scores dtype: {segment_scores.dtype}, shape: {segment_scores.shape}")

            # 创建一个新的IndexLSH来进行哈希分桶
            # print("Creating IndexLSH")
            # index_lsh = faiss.IndexLSH(d=1, nbits=nbits)
            index_lsh = faiss.IndexLSH(1, nbits)
            # print("IndexLSH created successfully")

            # Step 3.2：为当前段构建哈希分桶
            index_lsh.add(segment_scores)
            print("  Created IndexLSH and added segment scores.")
            # print("Added segment scores to IndexLSH")

            # Step 3.3：搜索桶ID
            # _, bucket_ids = index_lsh.search(segment_scores, k=1)
            # print(f"  Assigned Hash Buckets - bucket_ids shape: {bucket_ids.shape}")
            bucket_ids = index_lsh.assign(segment_scores)
            print(
                f"Assigned Hash Buckets - bucket_ids shape: {bucket_ids.shape}, unique buckets: {np.unique(bucket_ids)}")
            # print("Search in IndexLSH completed")

            # Step 3.4：按照哈希值将相似样本归入同一个桶
            buckets_for_segment = {}
            for j, bucket_id in enumerate(bucket_ids.ravel()):
                if bucket_id not in buckets_for_segment:
                    buckets_for_segment[bucket_id] = []
                buckets_for_segment[bucket_id].append(segment[j])

            # 将当前段的桶结果加入全局桶中
            buckets.append(buckets_for_segment)
        print("Step 3 Completed - Total Buckets Generated:", len(buckets))
        # print(f"Step 3 - Segment {i} Buckets:", buckets_for_segment)


        # Step 4: 桶内逐桶剪枝
        print("Starting Step 4 - Pruning within each bucket")
        pruned_indices = list(remained_indices)  # 初始化剩余的样本索引

        for group_idx, bucket_group in enumerate(buckets):
            print(f"Processing Bucket Group {group_idx + 1}/{len(buckets)}")
            for bucket_id, indices in bucket_group.items():
                print(f"  Bucket {bucket_id}: Sample Count = {len(indices)}")

                # 如果桶内有多个样本，保留最高分的样本
                if len(indices) > 1:
                    highest_score_index = indices[np.argmax(self.scores[indices])]
                    pruned_indices.append(highest_score_index)
                    print(f"    Retained Highest Score Sample Index: {highest_score_index}")

                    # 剩余样本进行按比例剪枝
                    num_to_keep = max(1, int(self.keep_ratio * (len(indices) - 1)))
                    selected_indices = np.random.choice(
                        [idx for idx in indices if idx != highest_score_index],
                        num_to_keep,
                        replace=False
                    )
                    pruned_indices.extend(selected_indices)
                    print(f"    Randomly Selected {num_to_keep} Samples for Retention: {selected_indices}")

                    # 按比例调整被选中样本的权重
                    self.weights[selected_indices] = 1 / self.keep_ratio
                else:
                    # 桶内只有一个样本，直接保留
                    pruned_indices.extend(indices)
                    print(f"    Single Sample in Bucket, Retained without Pruning: {indices[0]}")

        # 更新剪枝样本数
        self.num_pruned_samples += len(self.dataset) - len(pruned_indices)
        np.random.shuffle(pruned_indices)

        return pruned_indices

        # 原始的随机选择逻辑，用于后续步骤的对比测试
        # selected_indices = np.random.choice(
            # well_learned_indices, int(self.keep_ratio * len(well_learned_indices)), replace=False
        # )
        # self.reset_weights()
        # if len(selected_indices) > 0:
            # self.weights[selected_indices] = 1 / self.keep_ratio
            # remained_indices.extend(selected_indices)

        # self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        # np.random.shuffle(remained_indices)

        # return remained_indices

    '''


    '''
        def prune(self):
        mean_score = self.scores.mean().item()
        high_score_indices = np.where(self.scores > mean_score)[0]

        print(f"平均分数: {mean_score}")
        print(f"高于均值的样本索引: {high_score_indices}")

        # Step 1: 根据评分进行排序，按分数从大到小排列
        sorted_high_scores = high_score_indices[np.argsort(-self.scores[high_score_indices].numpy())]

        # Step 2: 分段，使用 alpha 指定的段数
        segment_size = len(sorted_high_scores) // self.alpha
        segments = [sorted_high_scores[i * segment_size:(i + 1) * segment_size] for i in range(self.alpha)]

        retained_indices = set()

        # Step 3: 针对每个段使用 LSH 分桶
        for segment in segments:
            if len(segment) == 0:
                continue  # 跳过空段
            data_to_encode = np.array([self.dataset[i][0].numpy() for i in segment])

            # 检查 data_to_encode 的形状并确保符合 LSH 需求
            print(f"data_to_encode shape before adjustment: {data_to_encode.shape}")

            if data_to_encode.shape[1] != self.d:
                print(f"Dimension mismatch: Expected feature dimension {self.d}, but got {data_to_encode.shape[1]}")
                continue  # 跳过维度不匹配的段

            print(f"data_to_encode shape after adjustment: {data_to_encode.shape}")
            self.index_lsh.reset()  # 重置哈希表
            self.index_lsh.add(data_to_encode)  # 重新添加数据

            _, labels = self.index_lsh.search(data_to_encode, k=len(segment))  # 获取每个样本的哈希桶
            buckets = {}

            for i, label_list in enumerate(labels):
                bucket_id = label_list[0]  # 假设第一个桶ID表示分配的哈希桶
                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(segment[i])

            # Step 4: 剪枝策略：保留每个桶中评分最高的样本，其他样本按照 prune_ratio 剪枝
            for bucket_id, bucket_samples in buckets.items():
                if len(bucket_samples) > 1:
                    sorted_samples = sorted(bucket_samples, key=lambda idx: -self.scores[idx].item())
                    retained_indices.add(sorted_samples[0])  # 保留最高分样本
                    pruned_samples = np.random.choice(sorted_samples[1:],
                                                      int(len(sorted_samples[1:]) * self.keep_ratio), replace=False)
                    retained_indices.update(pruned_samples)
                else:
                    retained_indices.update(bucket_samples)  # 样本少于等于1则不剪枝

        retained_indices = np.array(list(retained_indices))
        self.reset_weights()
        self.weights[retained_indices] *= 1 / self.keep_ratio
        return retained_indices
    '''





    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler



    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        #测试
        # print("-----------看看哪里有问题prune比较-----------", self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj) # may raise StopIteration

    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """
    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None

        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than InfoBatch.
        """
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj


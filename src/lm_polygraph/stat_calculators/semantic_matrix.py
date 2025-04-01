import numpy as np
from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
import torch.nn as nn
import torch
from tqdm import tqdm  # 添加进度条


class SemanticMatrixCalculator(StatCalculator):
    """
    计算生成样本的NLI语义矩阵，使用DeBERTa模型的内存优化版本。
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        返回计算器的统计数据和依赖项。
        """
        return [
            "semantic_matrix_entail",
            "semantic_matrix_contra",
            "semantic_matrix_classes",
            "semantic_matrix_entail_logits",
            "semantic_matrix_contra_logits",
            "entailment_id",
        ], ["sample_texts"]

    def __init__(self, nli_model, max_pairs_per_batch=256):
        super().__init__()
        self.is_deberta_setup = False
        self.nli_model = nli_model
        # 控制批处理大小以避免内存溢出
        self.max_pairs_per_batch = max_pairs_per_batch

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        使用DeBERTa模型计算生成样本的NLI语义矩阵，内存优化版本。

        Parameters:
            dependencies (Dict[str, np.ndarray]): 输入统计信息，包含：
                - 'sample_texts' (List[List[str]]): 批处理中每个输入文本的多个采样生成。
            texts (List[str]): 用于模型生成的输入文本批次。
            model (Model): 用于生成的模型。
            max_new_tokens (int): 模型生成的最大新令牌数。默认：100。
        Returns:
            Dict[str, np.ndarray]: 包含以下项目的字典：
                - 'semantic_matrix_entail' (List[np.array]): 每个输入文本的n_samples x n_samples大小的矩阵，
                  包含DeBERTa 'ENTAILMENT'输出的概率。
                - 'semantic_matrix_contra' (List[np.array]): 每个输入文本的n_samples x n_samples大小的矩阵，
                  包含DeBERTa 'CONTRADICTION'输出的概率。
                - 'semantic_matrix_entail_logits' (List[np.array]): 每个输入文本的n_samples x n_samples大小的矩阵，
                  包含DeBERTa 'ENTAILMENT'输出的logits。
                - 'semantic_matrix_contra_logits' (List[np.array]): 每个输入文本的n_samples x n_samples大小的矩阵，
                  包含DeBERTa 'CONTRADICTION'输出的logits。
                - 'semantic_matrix_classes' (List[np.array]): 每个输入文本的n_samples x n_samples大小的矩阵，
                  包含对应于DeBERTa预测的NLI标签ID。
        """
        # 设置基本组件
        deberta = self.nli_model
        device = deberta.device
        tokenizer = deberta.deberta_tokenizer
        ent_id = deberta.deberta.config.label2id["ENTAILMENT"]
        contra_id = deberta.deberta.config.label2id["CONTRADICTION"]
        softmax = nn.Softmax(dim=1)

        batch_texts = dependencies["sample_texts"]

        # 结果容器
        all_E = []
        all_C = []
        all_E_logits = []
        all_C_logits = []
        all_P = []

        # 处理每组文本
        for texts_idx, texts in enumerate(batch_texts):
            print(f"处理文本集 {texts_idx+1}/{len(batch_texts)}")

            # 找出唯一文本以减少计算量
            unique_texts, inv = np.unique(texts, return_inverse=True)
            unique_count = len(unique_texts)

            # 使用float16初始化矩阵以节省内存
            unique_E = np.zeros((unique_count, unique_count), dtype=np.float16)
            unique_C = np.zeros((unique_count, unique_count), dtype=np.float16)
            unique_E_logits = np.zeros(
                (unique_count, unique_count), dtype=np.float16)
            unique_C_logits = np.zeros(
                (unique_count, unique_count), dtype=np.float16)
            unique_P = np.zeros((unique_count, unique_count), dtype=np.int8)

            # 按行处理矩阵，避免一次性生成所有组合
            for i in tqdm(range(unique_count), desc=f"批次 {texts_idx+1}/{len(batch_texts)}"):
                # 按小批次处理每行，以避免内存溢出
                for j_start in range(0, unique_count, self.max_pairs_per_batch):
                    j_end = min(
                        j_start + self.max_pairs_per_batch, unique_count)

                    # 创建当前批次的文本对
                    first_texts = [unique_texts[i]] * (j_end - j_start)
                    second_texts = [unique_texts[j]
                                    for j in range(j_start, j_end)]

                    # 跳过空批次
                    if not first_texts or not second_texts:
                        continue

                    # 使用自动混合精度
                    with torch.amp.autocast(device_type='cuda'):
                        # 编码文本对
                        encoded = tokenizer.batch_encode_plus(
                            list(zip(first_texts, second_texts)),
                            padding=True,
                            truncation=True,
                            max_length=512,  # 限制序列长度
                            return_tensors="pt"
                        ).to(device)

                        # 无梯度推理
                        with torch.no_grad():
                            outputs = deberta.deberta(**encoded)
                            logits = outputs.logits

                        # 计算概率
                        probs = softmax(logits)

                        # 获取特定值并移至CPU
                        entail_probs = probs[:,
                                                ent_id].cpu().detach().numpy()
                        contra_probs = probs[:, contra_id].cpu(
                        ).detach().numpy()
                        entail_logits = logits[:,
                                                ent_id].cpu().detach().numpy()
                        contra_logits = logits[:, contra_id].cpu(
                        ).detach().numpy()
                        class_preds = probs.argmax(
                            dim=1).cpu().detach().numpy()

                        # 更新矩阵
                        for idx, j in enumerate(range(j_start, j_end)):
                            unique_E[i, j] = entail_probs[idx]
                            unique_C[i, j] = contra_probs[idx]
                            unique_E_logits[i, j] = entail_logits[idx]
                            unique_C_logits[i, j] = contra_logits[idx]
                            unique_P[i, j] = class_preds[idx]

            # 恢复完整矩阵并转换回原始精度
            full_E = unique_E[inv, :][:, inv].astype(np.float32)
            full_C = unique_C[inv, :][:, inv].astype(np.float32)
            full_E_logits = unique_E_logits[inv, :][:, inv].astype(np.float32)
            full_C_logits = unique_C_logits[inv, :][:, inv].astype(np.float32)
            full_P = unique_P[inv, :][:, inv].astype(np.int64)

            # 存储结果
            all_E.append(full_E)
            all_C.append(full_C)
            all_E_logits.append(full_E_logits)
            all_C_logits.append(full_C_logits)
            all_P.append(full_P)

        # 堆叠最终结果
        return {
            "semantic_matrix_entail": np.stack(all_E),
            "semantic_matrix_contra": np.stack(all_C),
            "semantic_matrix_entail_logits": np.stack(all_E_logits),
            "semantic_matrix_contra_logits": np.stack(all_C_logits),
            "semantic_matrix_classes": np.stack(all_P),
            "entailment_id": ent_id,
        }

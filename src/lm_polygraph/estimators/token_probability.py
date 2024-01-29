import numpy as np

from typing import Dict

from .estimator import Estimator


class MaximumTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaximumTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.max(log_likelihood) for log_likelihood in log_likelihoods])
    
class MaxMinGapTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaxMinGapTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([np.array(log_likelihood).max()-np.array(log_likelihood).min() for log_likelihood in log_likelihoods])

class MeanGapTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MeanGapTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([np.abs(np.array(log_likelihood)[1:] - np.roll(np.array(log_likelihood), 1)[1:]).mean() for log_likelihood in log_likelihoods])
    
class MaxGapTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaxGapTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([np.abs(np.array(log_likelihood)[1:] - np.roll(np.array(log_likelihood), 1)[1:]).max() for log_likelihood in log_likelihoods])
    
class WeightedTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "WeightedTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        unc = []
        for log_likelihood in log_likelihoods:
            weighted_prob = []
            for i, logp in enumerate(log_likelihood):
                if i > 0:
                    weights = np.array([(w+1)*2. for w in range(i)])
                    weights /= weights.sum()
                    weighted_prob.append(logp * (weights*log_likelihood[:i]).sum())
                else:
                    weighted_prob.append(logp)
            unc.append(np.sum(weighted_prob))
        return np.array(unc)
    
class Top03TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top03TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        topmean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argwhere(log_likelihood > np.log(0.3))
            if len(ind):
                topmean.append(-np.mean(log_likelihood[ind]))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top05TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top05TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        topmean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argwhere(log_likelihood > np.log(0.5))
            if len(ind):
                topmean.append(-np.mean(log_likelihood[ind]))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top07TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top07TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        topmean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argwhere(log_likelihood > np.log(0.7))
            if len(ind):
                topmean.append(-np.mean(log_likelihood[ind]))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top3TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top3TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top3mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(-log_likelihood)[:3] 
            top3mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top3mean)
    
class Top5TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top5TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top5mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(-log_likelihood)[:5] 
            top5mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top5mean)
    
class Top10TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top10TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top10mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(-log_likelihood)[:10] 
            top10mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top10mean)  
    
class Top50TokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top50TokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top50mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(-log_likelihood)[:50] 
            top50mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top50mean)  

    
class MinimumTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MinimumTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        return np.array([-np.min(log_likelihood) for log_likelihood in log_likelihoods])
    
    
class Top3MinTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top3MinTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top3mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(log_likelihood)[:3] 
            top3mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top3mean)
    
class Top5MinTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top5MinTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top5mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(log_likelihood)[:5] 
            top5mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top5mean)
    
class Top10MinTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top10MinTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top10mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(log_likelihood)[:10] 
            top10mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top10mean)  
    
class Top50MinTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "Top50MinTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_likelihoods = stats["greedy_log_likelihoods"]
        top50mean = []
        for log_likelihood in log_likelihoods:
            log_likelihood = np.array(log_likelihood)
            ind = np.argsort(log_likelihood)[:50] 
            top50mean.append(-np.mean(log_likelihood[ind]))
        return np.array(top50mean)  
    
    
class Top3MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top3MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :3] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top5MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top5MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :5] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
    
class Top10MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top10MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :10] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top50MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top50MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :50] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top100MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top100MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :100] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top1000MeanMeanTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top1000MeanMeanTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :1000] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.mean(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
    
class Top3MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top3MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :3] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top5MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top5MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :5] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
    
class Top10MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top10MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :10] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top50MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top50MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :50] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top100MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top100MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:100] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top1000MeanMaxTokenProbability(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top1000MeanMaxTokenProbability"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :1000] 
            top_i = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)
                top_i.append(np.mean(probi[mask]))
            if len(ind):
                topmean.append(-np.max(top_i))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top3MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top3MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :3] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top5MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top5MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :5] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top10MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top10MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :10] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)
    
class Top50MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top50MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :50] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)

class Top100MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top100MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :100] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)

class Top1000MeanTokenEntropy(Estimator):
    def __init__(self):
        super().__init__(["greedy_log_probs"], "sequence")

    def __str__(self):
        return "Top1000MeanTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["greedy_log_probs"]
        topmean = []
        for log_prob in log_probs:
            log_prob = np.array(log_prob)
            ind = (-log_prob).argsort(-1)[:, :1000] 
            entropies = []
            for i, prob in zip(ind, log_prob):
                probi = prob[i]
                mask = ~np.isinf(probi)                
                entropies.append(-np.mean(np.array(probi[mask]) * np.exp(probi[mask])))
            if len(ind):
                topmean.append(np.mean(entropies))
            else:
                topmean.append(1)
        return np.array(topmean)
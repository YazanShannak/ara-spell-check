import enum
from typing import List, Tuple


def precision(result: List[int], target: List[int], target_len: int, end_index: int) -> float:
    """
    Calculate precision.
    """

    pruned_target = target[1 : target_len - 1]
    pruned_result = result[1 : target_len - 1]

    correct = [token for index, token in enumerate(pruned_result) if token == pruned_target[index]]
    return len(correct) / len([token for token in result[1 : result.index(end_index)]])


def recall(result: List[int], target: List[int], target_len: int) -> float:
    """
    Calculate recall.
    """

    pruned_target = target[1 : target_len - 1]
    pruned_result = result[1 : target_len - 1]

    correct = [token for index, token in enumerate(pruned_result) if token == pruned_target[index]]

    return len(correct) / len(pruned_target)


def f1(precision: float, recall: float) -> float:
    """
    Calculate f1.
    """
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(
    result: List[int], target: List[int], end_index: int, target_len: int
) -> Tuple[float, float, float]:
    """
    Calculate metrics.
    """

    prec = precision(result=result, target=target, target_len=target_len, end_index=end_index)
    rec = recall(result=result, target=target, target_len=target_len)
    f = f1(precision=prec, recall=rec)

    return prec, rec, f

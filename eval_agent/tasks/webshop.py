import json
import logging
import random
from typing import List, Tuple, Any

logger = logging.getLogger("agent_frame")

from eval_agent.tasks.base import Task


class WebShopTask(Task):
    task_name = "webshop"

    def __init__(
        self,
        session_id: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
    
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1) -> Tuple[List[Task], int]:
        print(f"Loading {split} tasks")
        if split == 'train_1k':
            print("Loading 1000 train indices")
            idxs = json.load(open("eval_agent/data/webshop/train_indices.json"))
            idxs = idxs[:1000]
        elif split == 'train_2k' or split == 'train':
            print("Loading all 1824 train indices")
            idxs = json.load(open("eval_agent/data/webshop/train_indices.json"))

        elif split == 'test':
            print("Loading test indices")
            idxs = json.load(open("eval_agent/data/webshop/test_indices.json"))
        elif split == 'test_500':
            print("Loading test indices")
            idxs = json.load(open("eval_agent/data/webshop/test_indices_500.json"))
        else:
            raise NotImplementedError
        if part_num == 1:
            idxs = idxs
        else:
            assert part_idx != -1
            part_len = len(idxs) // part_num 
            idxs = idxs[part_len * part_idx: part_len * (part_idx + 1)]
        N_TASKS = len(idxs)
        def generator():
            for idx in idxs:
                session_id = idx
                yield cls(
                    task_id=idx,
                    session_id=session_id,
                )

        return generator(), N_TASKS
    

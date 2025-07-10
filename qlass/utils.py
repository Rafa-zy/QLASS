from copy import deepcopy
from typing import List, Dict, Any, Optional
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt

# need to deploy at local env.
def draw_graph_v2(G, pos, idx=0):
    # Use Graphviz layout to draw the tree
    pos = graphviz_layout(G, prog='dot')  # 'dot' engine gives a top-down layout
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=9, edge_color='gray', font_weight='bold')
    plt.savefig(f"tree_{idx}.png")
    plt.show()
    
def draw_graph(G, pos, idx=0):
    labels = {node: G.nodes[node]['label'] for node in G}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2500, node_color='skyblue', font_size=9, font_weight='bold', edge_color='gray')
    plt.savefig(f"tree_{idx}.png")
    plt.show()

class State:
    """This should contains everything needed to continue the conversation.

    For example, the history of the conversation, the current task (success/failure) at each step, etc.
    """

    def __init__(
        self,
        reward: float = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: str = None,
    ):
        """
        The history should be a format like:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        """
        self.history: List[Dict[str, Any]] = []
        self.reward: float = reward
        self.finished: bool = finished
        self.success: bool = success
        self.terminate_reason: str = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0

    @classmethod
    def load_json(cls, json_dict: Dict[str, Any]):
        state = cls()
        state.history = json_dict['conversations']
        info = json_dict['meta']
        state.reward = info["reward"]
        state.finished = info["finished"]
        state.success = info["success"]
        state.terminate_reason = info["terminate_reason"]
        state.error = info["error"]
        state.steps = info["steps"]
        return state

    @property
    def empty(self):
        return len(self.history) == 0

    def to_dict(self, format="fastchat") -> Dict[str, Any]:
        if format == 'openai':
            history = deepcopy(self.history)
        elif format == 'fastchat':
            history = []
            for idx, conv in enumerate(self.history):
                if idx % 2 == 0:
                    assert conv['role'] == 'user'
                    history.append({
                        "from": "human",
                        "value": conv['content'].strip(),
                    })
                else:
                    assert conv['role'] == 'assistant'
                    history.append({
                        "from": "gpt",
                        "value": conv['content'].strip(),
                    })
        else:
            raise NotImplementedError(f"Format {format} not implemented.")
        meta_info = {
            "steps": self.steps,
            "reward": self.reward,
            "finished": self.finished,
            "success": self.success,
            "terminate_reason": self.terminate_reason,
            "error": self.error,
        }
        res_dict = {
            "meta": meta_info,
            "conversations": history
        }
        return res_dict


# PROMPT_WITH_ICL_TEMPLATE = """{instruction}
# ---
# {icl_prompt}

# {examples}
# ---
# \nRemember that {instruction}\n
# Now, it's your turn and here is the task.<|user|>\n
# {task}<|assistant|>\n"""
PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
{icl_prompt}

{examples}
---
Now, it's your turn and here is the task.
{task}"""


PROMPT_WITHOUT_ICL_TEMPLATE = """{instruction}
Now, here is the task.
{task}"""

def prompt_without_icl(instruction, cur_task):
    prompt = PROMPT_WITHOUT_ICL_TEMPLATE.format(instruction=instruction, task=cur_task)
    messages = [{
        "role": "user",
        "content": instruction
    }]
    messages.append({
        "role": "assistant",
        "content": "OK"
    })
    messages.append({
        "role": "user",
        "content": cur_task
    })
    return prompt, messages

def prompt_with_icl(instruction, raw_icl, cur_task, icl_num=1):
    examples = ""
    messages = [{
        "role": "user",
        "content": instruction
    }]
    
    for i in range(min(icl_num, len(raw_icl))):
        exp = raw_icl[i]
        for j in range(len(exp)):
            cur_content = raw_icl[i][j]['content']
            if i == 0 and j == 0:
                messages.append({
                    "role": "assistant",
                    "content": "OK"
                })
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                if icl_num > 1:
                    examples += f"Example task {i + 1}:\n"
                examples += cur_content + '\n'
                continue
            elif i != 0 and j == 0:
                if icl_num > 1:
                    examples += f"\nExample task {i + 1}:\n"
                    examples += cur_content + '\n'
                else:
                    examples += '\n' + cur_content + '\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
                continue
            # user
            if j % 2 == 0:
                examples +=  cur_content + '\n\n'
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
            # assistant
            else:
                examples += cur_content + '\n'
                messages.append({
                    "role": "assistant",
                    "content": cur_content
                })
    icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(instruction=instruction, icl_prompt=icl_prompt, examples=examples, task=cur_task)
    messages.append({
        "role": "user",
        "content": cur_task
    })

    return prompt, messages

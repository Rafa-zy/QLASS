import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import logging
import os

class QNet(nn.Module):
    def __init__(self, hidden_size, args, accelerator=None, pad_token_id=None, no_load=False, mode='each',task_mode='lumos', model_args=None,training_args=None):
        super(QNet, self).__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.mode = mode
        if task_mode == 'lumos':
            if args.model_name_or_path and not no_load:
                self.llama = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
            else:
                logger = logging.getLogger(__name__)
                logger.info("Training new model from scratch")
                self.llama = AutoModelForCausalLM.from_config(config)
            
        elif task_mode == 'eto':
            self.llama = AutoModelForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        cache_dir=training_args.cache_dir,
                        trust_remote_code=model_args.trust_remote_code,
                        attn_implementation="flash_attention_2",
                        torch_dtype=torch.float16
                    )
            self.config = self.llama.config
        else:
            raise ValueError(f'Invalid task_mode: {task_mode}')
        self.config = self.llama.config
        self.config.pad_token_id = 0
        self.llama = self.llama.bfloat16()
        # Freeze the llama
        # for param in self.llama.parameters():
        #     param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1, bias=False)  # Q as a scalar
        ).bfloat16()
        self.apply_sigmoid = model_args.apply_sigmoid
        if self.apply_sigmoid:
            print('Applying sigmoid to the output')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.mlp(hidden_states)
        
        batch_size = input_ids.shape[0]
        
        if self.config.pad_token_id is None:
            if batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            sequence_lengths = -1
        else:
            # Compute sequence lengths accounting for padding
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
            # print(f'sequence_lengths:{sequence_lengths}')
        # print(sequence_lengths)
        if self.mode == 'each':
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), :sequence_lengths[0]]
        else:
            # Select logits at the positions determined by sequence_lengths
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        # print(f'pooled_logits:{pooled_logits.shape}')
        if self.apply_sigmoid:
            pooled_logits = torch.sigmoid(pooled_logits)
        return pooled_logits

    def save_pretrained(self, save_directory, is_main_process, save_function, state_dict=None):
        """Save the model components using Accelerate for distributed training compatibility.
        
        Args:
            save_directory (str): The directory to save the model's state dict and configuration.
            accelerator (Accelerator): The Accelerate library's Accelerator object, used to ensure 
                                    proper handling in distributed training environments.
        """
        # Ensure the directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Only the main process should perform the saving to prevent duplication
        if is_main_process:
            # Save using Accelerate's save function to be compatible with distributed environments
            save_function(state_dict, os.path.join(save_directory, "pytorch_model.bin"))


    @classmethod
    def from_pretrained(cls, load_directory, accelerator=None, args=None):
        """Load the model components, supporting loading in a distributed training setup if using Accelerate.
        
        Args:
            load_directory (str): The directory from which to load the model's state dict and configuration.
            accelerator (Accelerator, optional): The Accelerate library's Accelerator object. If provided,
                                                it's used to ensure proper handling in distributed training
                                                environments.
            args (argparse.Namespace, optional): Arguments needed for initializing the model.
        
        Returns:
            An instance of the model loaded with the pretrained weights.
        """
        # Load the configuration to determine model parameters
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(config_path)
            hidden_size = config.hidden_size  # example of pulling a needed param
        else:
            hidden_size = 4096  # default or error handling

        # Create an instance of the model with required initial parameters
        model = cls(hidden_size, args, accelerator, no_load=True)

        # Load the state dict
        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path)

        # If an Accelerator object is provided, use it to correctly load the state dict
        if accelerator:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        return model
    
    
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs):
        self.llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)





class QNet_V1(nn.Module):
    def __init__(self, hidden_size, args, accelerator):
        super(QNet_V1, self).__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)

        if args.model_name_or_path:
            self.llama = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
        else:
            logger = logging.getLogger(__name__)
            logger.info("Training new model from scratch")
            self.llama = AutoModelForCausalLM.from_config(config)
        self.config = self.llama.config
        self.llama = self.llama.bfloat16()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)  # Q as a scalar
        ).bfloat16()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).bfloat16()
        sum_hidden_state = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_attention_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_hidden_state = sum_hidden_state / sum_attention_mask
        q_value = self.mlp(pooled_hidden_state)
        return q_value

    def save_pretrained(self, save_directory, is_main_process, save_function, state_dict=None):
        """Save the model components using Accelerate for distributed training compatibility.
        
        Args:
            save_directory (str): The directory to save the model's state dict and configuration.
            accelerator (Accelerator): The Accelerate library's Accelerator object, used to ensure 
                                    proper handling in distributed training environments.
        """
        # Ensure the directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Only the main process should perform the saving to prevent duplication
        if is_main_process:
            # Save using Accelerate's save function to be compatible with distributed environments
            save_function(state_dict, os.path.join(save_directory, "pytorch_model.bin"))


    @classmethod
    def from_pretrained(cls, load_directory, accelerator=None, args=None):
        """Load the model components, supporting loading in a distributed training setup if using Accelerate.
        
        Args:
            load_directory (str): The directory from which to load the model's state dict and configuration.
            accelerator (Accelerator, optional): The Accelerate library's Accelerator object. If provided,
                                                it's used to ensure proper handling in distributed training
                                                environments.
            args (argparse.Namespace, optional): Arguments needed for initializing the model.
        
        Returns:
            An instance of the model loaded with the pretrained weights.
        """
        # Load the configuration to determine model parameters
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(config_path)
            hidden_size = config.hidden_size  # example of pulling a needed param
        else:
            hidden_size = 4096  # default or error handling

        # Create an instance of the model with required initial parameters
        model = cls(hidden_size, args, accelerator)

        # Load the state dict
        state_dict_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path)

        # If an Accelerator object is provided, use it to correctly load the state dict
        if accelerator:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        return model


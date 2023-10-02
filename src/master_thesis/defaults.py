from peft import LoraConfig
from transformers import TrainingArguments


DEFAULT_PEFT_CONFIG = LoraConfig(
    lora_alpha=16,  # Higher values assign more weight to the LoRA activations.
    lora_dropout=0.1,  # Dropout probability of the LoRA layers.
    r=64,  # The dimension of the low-rank matrices.
    bias="none",  # Whether to train all bias parameters or not.
)

DEFAULT_TRAIN_CONFIG = TrainingArguments(
    # Duration
    num_train_epochs=3,  # Total number of training epochs to perform.
    max_steps=-1,  #  If set to a positive number, the total number of training steps to perform.
    # Training
    group_by_length=True,  # Whether or not to group together samples of roughly the same length in the training dataset.
    gradient_accumulation_steps=1,  # Number of updates steps to accumulate before performing a backward/update pass.
    max_grad_norm=0.3,  # Maximum gradient norm (for gradient clipping).
    lr_scheduler_type="constant",  # The scheduler type to use.
    learning_rate=2e-4,  # Ratio of total training steps used for a linear warmup from 0 to learning_rate.
    optim="paged_adamw_32bit",  # The optimizer to use.
    warmup_ratio=0.03,  # Linear warmup over warmup_ratio fraction of total steps.
    weight_decay=0.001,  # The weight decay to apply (if not zero) to all layers except.
    # Hardware
    per_device_train_batch_size=4,  # Batch size per GPU/TPU core for training.
    fp16=False,  # Whether to use fp16 16-bit (mixed) precision training.
    bf16=False,  # Whether to use bf16 16-bit (mixed) precision training
    # Storage
    output_dir="../models",  # Directory where model predictions and checkpoints will be written.
    save_steps=25,  # Number of updates steps before two checkpoint saves.
    # Logging
    logging_steps=25,  # Number of update steps between two logs.
    report_to="all",  # The list of integrations to report the results and logs to.
)

RANDOM_SEED = 42

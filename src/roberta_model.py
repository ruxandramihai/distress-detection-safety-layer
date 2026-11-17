import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import random

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from datasets import Dataset as HFDataset, DatasetDict

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

LABEL_MAP = {
    'neutral': 0,
    'moderate': 1,
    'high-risk': 2
}

EMOTION_INDEX = {
    0: 'neutral',
    1: 'moderate',
    2: 'high-risk'
}

def load_dataset(clean=True):


    with open("data/train_data.json", "r") as f:
        train = json.load(f)
    with open("data/val_data.json", "r") as f:
        val = json.load(f)
    with open("data/test_data.json", "r") as f:
        test = json.load(f)

    train_df = pd.DataFrame(train["data"])
    val_df = pd.DataFrame(val["data"])
    test_df = pd.DataFrame(test["data"])

    if clean:
        train_len_before = len(train_df)
        val_len_before = len(val_df)
        test_len_before = len(test_df)

        train_df = train_df.dropna(subset=['text'])
        val_df = val_df.dropna(subset=['text'])
        test_df = test_df.dropna(subset=['text'])

        train_removed = train_len_before - len(train_df)
        val_removed = val_len_before - len(val_df)
        test_removed = test_len_before - len(test_df)

        print(f"Removed {train_removed} rows with NaN values from training dataset")
        print(f"Removed {val_removed} rows with NaN values from validation dataset")
        print(f"Removed {test_removed} rows with NaN values from testing dataset")
        print(f"Total removed: {train_removed + val_removed + test_removed} rows")

    return {'train': train_df, 'val': val_df, 'test': test_df}


class BERTDataset(Dataset):

    def __init__(self, df, tokenizer, max_length, datacolumn, labelcolumn):
        self.data = df
        self.texts = self.data[datacolumn].tolist()
        self.labels = self.data[labelcolumn].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if isinstance(label, str):
            label = LABEL_MAP[label]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }


def create_batch_tokenized_datasets_roberta(max_length=512, model_name="roberta-base"):
    dataframes = load_dataset(clean=True)
    hf_datasets = {}

    for split, df in dataframes.items():
        df = df.copy()
        df['label'] = df['my_label'].map(LABEL_MAP)
        hf_datasets[split] = HFDataset.from_pandas(df)

    dataset_dict = DatasetDict(hf_datasets)
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize(batch):
        return roberta_tokenizer(
            batch["text"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

    dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)
    dataset_encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    print(f"Created batch-tokenized datasets:")
    for split, dataset in dataset_encoded.items():
        print(f"  {split}: {len(dataset)} examples")

    return dataset_encoded


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.mean(predictions == labels)

    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
    }

def setup_training_args():
    return TrainingArguments(
        output_dir='./models/resultsroberta',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        metric_for_best_model="f1",
        disable_tqdm=False,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        no_cuda=True,
        save_strategy="epoch",
        use_mps_device=False,
        logging_dir='./logs',
        logging_steps=10,
        report_to=None,
        seed=42,
    )

def plot_training_history(trainer):

    log_history = trainer.state.log_history
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    train_losses = [log['loss'] for log in train_logs if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in eval_logs if 'eval_loss' in log]


    train_epochs = [log['epoch'] for log in train_logs if 'loss' in log]
    eval_epochs = [log['epoch'] for log in eval_logs if 'eval_loss' in log]
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    if train_losses and eval_losses:
        plt.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(eval_epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    eval_accuracies = [log['eval_accuracy'] for log in eval_logs if 'eval_accuracy' in log]
    if eval_accuracies:
        plt.plot(eval_epochs, eval_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./models/resultsroberta/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):

    labels = ['neutral', 'moderate', 'high-risk']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def main():

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name_roberta = "roberta-base"
    num_labels = 3

    roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name_roberta)
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        model_name_roberta,
        num_labels=num_labels
    ).to(device)

    dataset_encoded_roberta = create_batch_tokenized_datasets_roberta(
        max_length=512,
        model_name="roberta-base"
    )

    training_args = setup_training_args()

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001
    )

    trainer_roberta = Trainer(
        model=roberta_model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded_roberta["train"],
        eval_dataset=dataset_encoded_roberta["val"],
        callbacks=[early_stopping]
    )

    trainer_roberta.train()

    roberta_model.save_pretrained('./models/resultsroberta')
    roberta_tokenizer.save_pretrained('./models/resultsroberta')

    preds_output_roberta = trainer_roberta.predict(dataset_encoded_roberta["test"])
    print("Test metrics:", preds_output_roberta.metrics)

    y_valid_roberta = np.array(dataset_encoded_roberta["test"]["label"])
    y_preds_roberta = np.argmax(preds_output_roberta.predictions, axis=1)


    plot_confusion_matrix(
        y_valid_roberta,
        y_preds_roberta,
        title='Confusion Matrix RoBERTa'
    )

    print('\nClassification Report for RoBERTa:\n')
    print(classification_report(
        y_valid_roberta,
        y_preds_roberta,
        target_names=['neutral', 'moderate', 'high-risk']
    ))


if __name__ == "__main__":
    main()

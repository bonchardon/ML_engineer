# todo: language detection (Turkish, Azerbaijani, Swahili), so that the quality of language
#  detection received through 'SIP-телефонию (кодек G.729)' would be higher
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import load_dataset, concatenate_datasets


def lang_rec():
    # Step 1: Prepare data (not included in this example)
    # Load datasets for Turkish, Azerbaijani, and Swahili from Common Voice
    GET_ENV = os.getenv('use_auth_token')
    common_voice_tr = load_dataset("mozilla-foundation/common_voice_13_0", "tr", split="train", use_auth_token=GET_ENV)
    common_voice_az = load_dataset("mozilla-foundation/common_voice_13_0", "az", split="train", use_auth_token=GET_ENV)
    common_voice_sw = load_dataset("mozilla-foundation/common_voice_13_0", "sw", split="train", use_auth_token=GET_ENV)

    # Merging datasets
    combined_dataset = concatenate_datasets([common_voice_tr, common_voice_az, common_voice_sw])

    # Splitting the combined dataset into train and test sets
    train_dataset, test_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    #  Step #2: we are using the facebook/wav2vec2-large-xlsr-53 model,
    #  which is trained on a wide range of languages, including Turkish, Azerbaijani, and Swahili.
    model_name = "facebook/wav2vec2-large-xlsr-53"
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Step 3: Define and prepare the model for fine-tuning
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model.freeze_feature_extractor()

    # Step 4: Fine-tune the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Step 5: training part!
    train_dataset = Dataset(train_dataset, tokenizer, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CTCLoss(blank_id=processor.tokenizer.pad_token_id)

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            input_values = processor(batch["input_values"], return_tensors="pt", padding=True,
                                     truncation=True).input_values.to(device)
            labels = batch["labels"]
            with torch.no_grad():
                logits = model(input_values).logits

            labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            logits = logits.permute(1, 0, 2)
            input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long).to(device)
            label_lengths = torch.full((logits.shape[1],), labels.shape[1], dtype=torch.long).to(device)

            optimizer.zero_grad()
            loss = loss_fn(logits, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

    # Step 5: Evaluate the fine-tuned model
    test_data = load_dataset("common_voice", "tr", split="test")
    test_dataset = Dataset(test_data, tokenizer, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_values = processor(batch["input_values"], return_tensors="pt", padding=True,
                                     truncation=True).input_values.to(device)
            labels = batch["labels"]
            logits = model(input_values).logits
            labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            logits = logits.permute(1, 0, 2)
            input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long).to(device)
            label_lengths = torch.full((logits.shape[1],), labels.shape[1], dtype=torch.long).to(device)
            loss = loss_fn(logits, labels, input_lengths, label_lengths)
            total_loss += loss.item()

    print("Test Loss:", total_loss / len(test_loader))


if __name__ == "__main__":

    lang_rec()

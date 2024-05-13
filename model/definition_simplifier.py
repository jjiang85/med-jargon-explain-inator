import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
import pandas as pd

class MedicalDataset(Dataset):

    def __init__(self, sentences, simplified_sentences, max_length, tokenizer):
        self.sentences = sentences
        self.simplified_sentences = simplified_sentences
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        simplified_sentence = self.simplified_sentences[index]

        encoding = self.tokenizer.encode_plus(
                                sentence,
                                simplified_sentence,
                                add_special_tokens=True,
                                max_length=self.max_length,
                                padding='max_length',
                                return_tensors='pt',
                                truncation=True)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1)  # Dummy label, since this is for simplification not classification
                }

def load_dataset(filepath):
    """
    Loads a .csv file into a pandas dataframe, then converts the information from each column
    into a list for ease of access later in the model

    args:
        filepath (str): indicates the relative file path and location to access the dataset

    returns:
        sentences (list): sentence strings from the wikipedia aligned dataset
        simplified_sentences (list): corresponding simplified sentence strings from the wikipedia aligned dataset
    """
    
    df = pd.read_csv(filepath)
    sentences = df['sentence'].to_list()
    simplified_sentences = df['simple_sentence'].to_list()

    return sentences, simplified_sentences

def simplify_sentence(sentence, model, tokenizer, max_length=128, beam_search=False, beam_size=3):
    """
    encodes a given sentence, runs it through the simplification model, then decodes it to return a simplified sentence

    args:
        sentence (str): The sentence to be simplified.
        model: The pre-trained simplification model.
        tokenizer: The tokenizer used for encoding the input sentence.
        max_length (int): The maximum length of the input sequence after tokenization, defaults to 128.
        beam_search (bool): Whether to use beam search decoding, defaults to False.
        beam_size (int): The beam size for beam search decoding, applicable only if beam_search is True, defaults to 3.

    returns:
        simplified_sentence (str): The simplified version of the input sentence provided
    """
    inputs = tokenizer.encode_plus(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, 
                                  num_beams=beam_size if beam_search else 1, early_stopping=True)
    
    simplified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified_sentence



if __name__ == '__main__':

    filepath = 'data/def_simplifier_training/aligned_data.csv'

    sentences, simplified_sentences = load_dataset(filepath)

    # Find the longest sentence to ensure proper padding length
    long_sent = max(sentences, key=lambda x: len(x))
    long_simp_sent = max(simplified_sentences, key=lambda x: len(x))
    max_length = max(long_sent, long_simp_sent)

    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Define parameters
    batch_size = 8
    max_length = 128
    learning_rate = 2e-5
    epochs = 2

    # Prepare the dataset and data loader for the model
    train_ds = MedicalDataset(sentences, simplified_sentences, max_length, tokenizer)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader)}')

    # Save the fine-tuned model and tokenizer
    output_dir = "fine_tuned_medical_bert"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Example usage of simplifying sentences
    sents = ["A neurotransmitter (a chemical messenger that sends signals between brain cells) that plays roles in attention, learning, and memory.",
                 "The patient presented with a myocardial infarction."]

    with open('output.txt', "w") as f:
        for sent in sents:
            simp_sent = simplify_sentence(sent, model, tokenizer)
            f.write("Original sentence:" + '\n' + str(sent))
            f.write("Simplified sentence:" + '\n' + str(simp_sent) + '\n')



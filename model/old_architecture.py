import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

# def simplify_sentence(sentence, model, tokenizer, max_length=128, beam_search=False, beam_size=3):
#     """
#     encodes a given sentence, runs it through the simplification model, then decodes it to return a simplified sentence

#     args:
#         sentence (str): The sentence to be simplified.
#         model: The pre-trained simplification model.
#         tokenizer: The tokenizer used for encoding the input sentence.
#         max_length (int): The maximum length of the input sequence after tokenization, defaults to 128.
#         beam_search (bool): Whether to use beam search decoding, defaults to False.
#         beam_size (int): The beam size for beam search decoding, applicable only if beam_search is True, defaults to 3.

#     returns:
#         simplified_sentence (str): The simplified version of the input sentence provided
#     """
#     inputs = tokenizer.encode_plus(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']

#     with torch.no_grad():
#         outputs = model.generate(input_ids=input_ids, 
#                                 attention_mask=attention_mask, 
#                                 max_length=max_length, 
#                                 num_beams=beam_size if beam_search else 1, 
#                                 early_stopping=True)
    
#     simplified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return simplified_sentence

# BART
# def simplify_sentence(sentence, model, tokenizer):
#     inputs = tokenizer.encode("simplify: " + sentence, return_tensors="pt", max_length=1024, truncation=True)
#     outputs = model.generate(inputs, max_length=1024, num_beams=4, early_stopping=True)
#     simplified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return simplified_sentence

# T5
def simplify_sentence(sentence, model, tokenizer):
    # Prepend the task prefix for T5
    input_text = "simplify: " + sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    simplified_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified_sentence


if __name__ == '__main__':
    train = False
    if train == True:
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
    
    else:
        # BERT
        # model_name = "./fine_tuned_medical_bert"
        # model = BertForSequenceClassification.from_pretrained(model_name)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # BART
        # model_name = "facebook/bart-large-cnn"
        # model = BartForConditionalGeneration.from_pretrained(model_name)
        # tokenizer = BartTokenizer.from_pretrained(model_name)

        #T5
        model_name = "t5-base" 
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Example usage of simplifying sentences
        definitions = {"adhesion": "A band of scar tissue that joins normally separated internal body structures, most often after surgery, inflammation, or injury in the area.",
                        "arthroscopy": "A minimally invasive diagnostic and treatment procedure used for conditions of a joint. This procedure uses a small, lighted, optic tube (arthroscope) which is inserted into the joint through a small incision in the joint. Images of the inside of the joint are projected onto a screen; used to evaluate any degenerative and/or arthritic changes in the joint; to detect bone diseases and tumors; to determine the cause of bone pain and inflammation.",
                        "carpal tunnel": "Passageway in the wrist through which nerves and the flexor muscles of the hands pass.",
                        "CT scan": "Computed Tomography (CT) is a non-invasive, diagnostic procedure that uses a series of x-rays to show a cross-sectional view of the inside of the body.",
                        "fusion": "Correction of an unstable part of the spine by joining two or more vertebrae. Usually done surgically, but sometimes done by traction or immobilization.",
                        "gastroenterostomy": "Surgical creation of an opening between the stomach wall and the small intestines; performed when the normal opening has been eliminated.",
                        "meniscus": "Crescent-shaped cartilage between the upper end of the tibia (shin bone) and the femur (thigh bone).",
                        "pelvic floor": "Muscles and connective tissue providing support for pelvic organs; e.g. bladder, lower intestines, uterus (in females); also aids in continence as part of the urinary and anal sphincters.",
                        "pleurisy": "Inflammation of the pleura that is characterized by sudden onset, painful and difficult respiration and exudation of fluid or fibrinous material into the pleural cavity." 
                    }
        
        with open('T5output.txt', "w") as f:
            for key, value in definitions.items():
                simp_sent = simplify_sentence(value, model, tokenizer)
                f.write(key + "\n")
                f.write("Original sentence:" + '\n' + str(value) + '\n')
                f.write("Simplified sentence:" + '\n' + str(simp_sent) + '\n\n')



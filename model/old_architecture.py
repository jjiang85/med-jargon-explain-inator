import torch
import pandas as pd
import nltk

from datasets import load_metric, load_dataset, concatenate_datasets, DatasetDict
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import EncoderDecoderModel
from transformers import BertTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


class DefinitionSimplifier:

    def __init__(self):
        
        self.inputpath = '../data/def_simplifier_training/normal.aligned'
        self.outputpath = '../data/def_simplifier_training/simple.aligned'
        self.datapath = '../data/def_simplifier_training/aligned_data.csv'

    
    def load_BERT_model(self):

        # Initialize a pretrained model and tokenizer, loade from transformers
        model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Configure model embeddings using the tokenizer
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.max_length = 120
        model.config.min_length = 40
        model.config.early_stopping = True
        model.config.length_penalty = 0.8
        model.config.num_beams = 3

        # Save the model and tokenizer
        self.model = model
        self.tokenizer = tokenizer 


    def load_Wiki_data(self):
        dataset = load_dataset('csv', data_files=self.datapath, delimiter=',')
        dataset = dataset["train"].train_test_split(test_size=0.2)
        # print(dataset)

        train_ds = dataset['train'].shuffle(seed=42)
        eval_ds = dataset['test']

    
        # input_train = train_ds['sentence']
        # output_train = train_ds['simple_sentence']
        # train_df = DatasetDict({'input_ids': input_train, 'labels': output_train})
        # self.train_df = train_df

        # input_eval = eval_ds['sentence']
        # output_eval = eval_ds['simple_sentence']
        # eval_df = DatasetDict({'input_ids': input_eval, 'labels': output_eval})
        # self.eval_df = eval_df

        # print(train_ds)
        input_train = self.tokenizer.batch_encode_plus(train_ds['sentence'], padding=True, truncation=True, return_tensors='pt')
        output_train = self.tokenizer.batch_encode_plus(train_ds['simple_sentence'], padding=True, truncation=True, return_tensors='pt')
        train_df = DatasetDict({'input_ids': input_train, 'labels': output_train})
        self.train_df = train_df
        print(train_df)

        input_eval = self.tokenizer.batch_encode_plus(eval_ds['sentence'], padding=True, truncation=True, return_tensors='pt')
        output_eval = self.tokenizer.batch_encode_plus(eval_ds['simple_sentence'], padding=True, truncation=True, return_tensors='pt')
        eval_df = DatasetDict({'input_ids': input_eval, 'labels': output_eval})
        self.eval_df = eval_df

        self.dataset = {'train': train_df, 'eval': eval_df}
        # DatasetDict({'train': train_df,
        #                             'eval': eval_df})
        # print(self.dataset)



    def train_model(self):

        # Initialize and save training arguments for the model
        training_arguments = Seq2SeqTrainingArguments(
                            predict_with_generate=True,
                            evaluation_strategy='steps',
                            per_device_train_batch_size=8,
                            per_device_eval_batch_size=8,
                            fp16=torch.cuda.is_available(),
                            output_dir='./out',
                            logging_steps=100,
                            save_steps=3000,
                            eval_steps=10000,
                            warmup_steps=2000,
                            gradient_accumulation_steps=1,
                            save_total_limit=3,
                            remove_unused_columns= False)
        self.training_args = training_arguments
        

        meteor = load_metric('meteor')
        rouge = load_metric('rouge')

        # Define the compute metrics function required for Seq2Seq Training
        def compute_metrics(prediction):
            labels_ids = prediction.label_ids
            pred_ids = prediction.predictions

            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
            label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            meteor_output = meteor.compute(predictions=pred_str, references=label_str)
            rouge_output = rouge.compute(
                predictions=pred_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid

            return {'meteor_score': round(meteor_output['meteor'], 4),
                    'rouge2_precision': round(rouge_output.precision, 4),
                    'rouge2_recall': round(rouge_output.recall, 4),
                    'rouge2_f_measure': round(rouge_output.fmeasure, 4)}

    
        # Define the Seq2Seq Trainer
        trainer = Seq2SeqTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=training_arguments,
                    compute_metrics=compute_metrics,
                    train_dataset=self.dataset['train']['input_ids'],
                    eval_dataset=self.dataset['eval']['input_ids'])
                    # train_dataset=self.train_df['input_ids'],
                    # eval_dataset=self.eval_df['input_ids'])
                    # train_dataset=self.dataset['train']['input'],
                    # eval_dataset=self.dataset['eval']['input'])
                    # train_dataset=self.train,
                    # eval_dataset=self.eval)
                    # train_dataset=self.train_dataset,
                    # eval_dataset=self.eval_dataset)

        # Train the text simplification model
        trainer.train()
        trainer.save_model('../trained_models/saved_model')


    def evaluate_model(self, sentence):

        trained_model = EncoderDecoderModel.from_pretrained('../trained_models/saved_model')
        tokenizer = BertTokenizer.from_pretrained('../trained_models/saved_model')

        inputs = tokenizer([sentence], padding='max_length',
                            max_length=60, truncation=True, return_tensors='pt')

        trained_model.config.decoder_start_token_id = tokenizer.cls_token_id
        trained_model.config.eos_token_id = tokenizer.sep_token_id
        trained_model.config.pad_token_id = tokenizer.pad_token_id
        trained_model.config.vocab_size = self.model.config.encoder.vocab_size

        output = trained_model.generate(inputs['input_ids'],
                                max_length=60,
                                min_length=30,
                                num_beams=4,
                                length_penalty=0.8,
                                temperature=1.0,
                                early_stopping=True,
                                top_k=50,
                                do_sample=False)

        text = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(text)


if __name__ == '__main__':
    # Initialize the class
    simplifier = DefinitionSimplifier()

    # Load the model and training data
    simplifier.load_BERT_model()
    simplifier.load_Wiki_data()

    # Train the model and tune using Wiki Data
    simplifier.train_model()

    # Create sentence to be simplified
    sentence = "A neurotransmitter (a chemical messenger that sends signals between brain cells) that plays roles in attention, learning, and memory."
    # Test the model
    simplifier.evaluate_model(sentence)


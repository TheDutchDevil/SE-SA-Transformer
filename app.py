# Created by happygirlzt
# -*- coding: utf-8 -*-
from calendar import EPOCH
import sys
sys.path.append('/media/DATA/tingzhang-data/sa4se/scripts')

import numpy as np

from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import argparse
import pprint
import math
from transformers import AdamW
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import ProgressBar

import logging
logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser()

    # This script has three modes: Train, Predict, and List
    parser.add_argument("--train", dest="train", action="store_true")

    parser.add_argument("--predict", dest="predict", action="store_true")

    parser.add_argument("--list", dest="list", action="store_true")

    # Input and output files. 

    parser.add_argument("--input", dest="input", help="", type=str)

    parser.add_argument("--output", dest="output", help="", type=str)

    # The two columns can be used to deal with non-standard dataset files. 

    parser.add_argument("--model-name", dest="model_name", help="", type=str)

    parser.add_argument("--text-column", dest="text_column", help="", type=str, default="text")

    parser.add_argument("--label-column", dest="label_column", help="", type=str, default="label")

    parser.add_argument("--seed", dest="seed", help="Seed for the stratified split", type=int, default=None)

    parser.add_argument("--test-split", dest="test_split", help="Portion of the train set that will be used for the test split", type=float, default=0.3)

    args = parser.parse_args()

    # If more than one of the three modes is true, then fail with an error
    if sum([args.train, args.predict, args.list]) > 1:
        raise ValueError("Only one of --train, --predict, or --list can be true")

    input_file_name = args.input

    if args.train:
        train(input_file_name, text_column=args.text_column, label_column=args.label_column, model_name = args.model_name, stratified_seed=args.seed, test_split_portion = args.test_split)

    if args.predict:
        predict(input_file_name, args.output, text_column=args.text_column, label_column=args.label_column, model_name = args.model_name)

def train(file_name, text_column = "text", label_column = "polarity", stratified_seed = None, model_name = "default", test_split_portion = 0.3):

    if not Path(file_name).exists():
        raise ValueError(f"No input file found at {file_name}")


    input_df = pd.read_csv(file_name)

    input_df.columns= input_df.columns.str.lower()

    if "id" not in input_df.columns:
        input_df["id"] = input_df.index

    if text_column not in input_df.columns:
        raise ValueError(f"No text column named {text_column} found in input file")
    
    if label_column not in input_df.columns:
        raise ValueError(f"No label column named {label_column} found in input file")

    X, y = input_df[['id', text_column]], input_df[label_column]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_split_portion, stratify=y, random_state=stratified_seed)

    encoder = LabelEncoder()

    train_y = encoder.fit_transform(train_y)
    test_y = encoder.transform(test_y)

    m_num=0
    rerun_flag=True
        

    cur_model=MODELS[m_num]
    m_name=MODEL_NAMES[m_num]

    train_X_sentences = train_X[text_column]
    test_X_sentences = test_X[text_column]

    print('Read success!')

    # pred_iterator=get_iterator(X_test, y_test, cur_model, False)

    prediction_dataloader=get_dataloader(test_X_sentences, test_y, cur_model, False)

    # print('Training set is {}\nValidation set is {}\nTest set is {}'.format(len(train_dataloader.dataset), len(validation_dataloader.dataset), len(prediction_dataloader.dataset)))

    X_train, X_validation, y_train, y_validation = train_test_split(train_X_sentences, 
                                                            train_y, 
                                                            test_size=0.05, 
                                                            random_state=stratified_seed,
                                                            stratify=train_y)

    #train_dataloader=get_dataloader(X_train, y_train,cur_model,True)
    #validation_dataloader=get_dataloader(X_validation, y_validation,cur_model,False)

    train_iterator=get_iterator(X_train, y_train, cur_model)
    valid_iterator=get_iterator(X_validation, y_validation, cur_model)

    model = cur_model[0].from_pretrained(cur_model[2], num_labels=len(encoder.classes_))

    if torch.cuda.is_available():
        model.cuda()

    optimizer = AdamW(model.parameters(),
                        lr=LEARNING_RATE,
                        eps=EPS,
                        weight_decay=WEIGHT_DECAY)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8) # 5e-5 * 0.8 = 4e-5

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
            

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        logits = outputs[1]

        loss.backward()
        optimizer.step()
        #scheduler.step()
        return loss.item()

    def eval_function(engine, batch):
        model.eval()
        with torch.no_grad():
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            #logits = outputs[0]
            y_pred=outputs[0]
            
            return y_pred, b_labels

    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    #print('success!')
    #### Metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    def output_transform_fun(output):
        y_pred, y = output
        y_pred=y_pred.detach().cpu().numpy()
        y=y.to('cpu').numpy()
        y_pred=np.argmax(y_pred, axis=1).flatten()
        return torch.from_numpy(y_pred), torch.from_numpy(y)

    criterion = nn.CrossEntropyLoss()
    ### Training
    #Accuracy(output_transform=output_transform_fun).attach(train_evaluator, 'accuracy')
    Loss(criterion).attach(train_evaluator, 'cross-entropy')

    precision = Precision(output_transform=output_transform_fun, average=False)
    #.detach().cpu().numpy()
    recall = Recall(output_transform=output_transform_fun, average=False)
    #.detach().cpu().numpy()
    F1 = (precision * recall * 2) / (precision + recall)

    #precision.attach(train_evaluator, 'precision')
    #recall.attach(train_evaluator, 'recall')
    #F1.attach(train_evaluator, 'F1')

    ### Validation    
    #Accuracy(output_transform=output_transform_fun).attach(validation_evaluator, 'accuracy')
    Loss(criterion).attach(validation_evaluator, 'cross-entropy')

    #precision.attach(validation_evaluator, 'precision')
    #recall.attach(validation_evaluator, 'recall')
    #F1.attach(validation_evaluator, 'F1')

    #### Progress Bar
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])

    def score_function_loss(engine):
        val_loss = engine.state.metrics['cross-entropy']
        return -val_loss

    def score_function_f1(engine):
        val_f1 = engine.state.metrics['F1']
        if math.isnan(val_f1):
            return -9999
        return val_f1

    handler = EarlyStopping(patience=2, score_function=score_function_loss, trainer=trainer)

    validation_evaluator.add_event_handler(Events.COMPLETED, handler)

    def log_training_results(engine):
        train_evaluator.run(train_iterator)
        metrics = train_evaluator.state.metrics
        pbar.log_message(
        "Training Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))

    def log_validation_results(engine):
        validation_evaluator.run(valid_iterator)
        metrics = validation_evaluator.state.metrics
        pbar.log_message(
        "Validation Results - Epoch: {} \nMetrics\n{}"
        .format(engine.state.epoch, pprint.pformat(metrics)))
        pbar.n = pbar.last_print_n = 0
        
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

    #### Checkpoint

    # to_save = {'{}_{}'.format(p_name, m_name): model,
    #           'optimizer': optimizer,
    #           'lr_scheduler': scheduler
    #           }

    to_save={'gh_{}'.format(m_name): model}

    cp_handler = Checkpoint(to_save,
                        DiskSaver('models/',
                        create_dir=True, require_empty=False),
                        filename_prefix='best',
                        score_function=score_function_loss,
                        score_name='val_loss')

    #validation_evaluator.add_event_handler(Events.COMPLETED, cp_handler)
    #trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), cp_handler)

    # checkpointer = ModelCheckpoint('../models/', '{}'.format(p_name), create_dir=True, save_as_state_dict=True, require_empty=False)

    # trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.run(train_iterator, max_epochs=EPOCHS)


    predictions = run_model(prediction_dataloader, model)

    output_df = pd.DataFrame({'Prediction': predictions, 'GroundTruth': test_y, 'Text': test_X_sentences})


    output_df['Prediction'] = encoder.inverse_transform(output_df['Prediction'])
    output_df['GroundTruth'] = encoder.inverse_transform(output_df['GroundTruth'])

    
    print(classification_report(output_df['GroundTruth'], output_df['Prediction']))

    model.save_pretrained(f'models/bert_{model_name}/')

    np.save(f'models/encoder_{model_name}.npy', encoder.classes_)


def predict(file_name, output_file, model_name, text_column = "text", label_column = "label"):
    df_predict = pd.read_csv(file_name)


    df_predict.columns= df_predict.columns.str.lower()

    if "id" not in df_predict.columns:
        df_predict["id"] = df_predict.index

    if text_column not in df_predict.columns:
        raise ValueError(f"No text column named {text_column} found in input file")

    X = df_predict[['id', text_column]]

    encoder = LabelEncoder()
    encoder.classes_ = np.load(f'models/encoder_{model_name}.npy', allow_pickle=True)

    m_num=0
    
    cur_model=MODELS[m_num]

    sentences = X[text_column]
    
    prediction_dataloader = get_dataloader(sentences, None, cur_model, False)

    model = cur_model[0].from_pretrained(f'models/bert_{model_name}', num_labels=len(encoder.classes_), local_files_only = True)

    if torch.cuda.is_available():
        model.cuda()

    predictions = run_model_without_ground_truth(prediction_dataloader, model)

    df_predict[label_column] = encoder.inverse_transform(predictions)

    df_predict.to_csv(output_file, index=False)

    print("Saved output file to {}".format(output_file))

if __name__ == '__main__':
    main()



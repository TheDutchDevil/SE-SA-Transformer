# Created by happygirlzt
# -*- coding: utf-8 -*-
from calendar import EPOCH
import sys
sys.path.append('/media/DATA/tingzhang-data/sa4se/scripts')

from utils import *
from sklearn.model_selection import train_test_split
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

parser = argparse.ArgumentParser()

#parser.add_argument("--do-train", dest="do_train", help="Run a 70/30 train test split.", action="store_true")

parser.add_argument("--dataset", help='Path for the dataset. Should be a csv in the format id (optional, otherwise index is used), text, polarity')

parser.add_argument("--stratified-seed", dest="stratified_seed", help="Seed for the stratified split", type=int, default=None)

parser.add_argument("--run-name", dest="run_name", help="Name of the run", type=str)

parser.add_argument("--save-preds", dest="save_preds", help="whether the predictions should be saved to ./predictions-{run-name}.csv", action="store_true")

args = parser.parse_args()

input_file_name = args.dataset

if not Path(input_file_name).exists():
    raise ValueError(f"No input file found at {input_file_name}")


input_df = pd.read_csv(input_file_name)

if "id" not in input_df.columns:
    input_df["id"] = input_df.index



X, y = input_df[['id', 'text']], input_df['polarity']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.stratified_seed)

train_y = train_y.replace({'positive': 1, 'neutral': 0, 'negative': 2})
test_y = test_y.replace({'positive': 1, 'neutral': 0, 'negative': 2})

m_num=0
rerun_flag=True
    

cur_model=MODELS[m_num]
m_name=MODEL_NAMES[m_num]

train_X_sentences = train_X["text"]
test_X_sentences = test_X["text"]

print('Read success!')

# pred_iterator=get_iterator(X_test, y_test, cur_model, False)

prediction_dataloader=get_dataloader(test_X_sentences, test_y, cur_model, False)

# print('Training set is {}\nValidation set is {}\nTest set is {}'.format(len(train_dataloader.dataset), len(validation_dataloader.dataset), len(prediction_dataloader.dataset)))

X_train, X_validation, y_train, y_validation = train_test_split(train_X_sentences, 
                                                        train_y, 
                                                        test_size=0.05, 
                                                        random_state=args.stratified_seed,
                                                        stratify=train_y)

#train_dataloader=get_dataloader(X_train, y_train,cur_model,True)
#validation_dataloader=get_dataloader(X_validation, y_validation,cur_model,False)

train_iterator=get_iterator(X_train, y_train, cur_model, True)
valid_iterator=get_iterator(X_validation, y_validation, cur_model, False)

model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)

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
                    DiskSaver('../models/',
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


output_df['Prediction']=output_df['Prediction'].replace({1:'positive', 2:'negative', 0:'neutral'})
output_df['GroundTruth']=output_df['GroundTruth'].replace({1:'positive', 2:'negative', 0:'neutral'})

output_df.to_csv(f'{args.run_name}.csv', index=False)

print(f"Saved to {args.run_name}.csv")



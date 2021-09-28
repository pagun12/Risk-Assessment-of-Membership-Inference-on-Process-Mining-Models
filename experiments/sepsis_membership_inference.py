import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import time
import os
import sys
from sys import argv
import pickle
import joblib
import random
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

dataset_manager_sepsis = DatasetManager("sepsis_cases_1")
sepsis_data = dataset_manager_sepsis.read_dataset()

# features for classifier

# event attributes
dynamic_cols = ["Activity", 'CRP', 'Leucocytes', 'LacticAcid']

# case attributes that are known from the start
static_cols = ['Diagnose', 'Age']

# case attributes that are bool values
static_cols_bool = ['DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                    'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                    'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                    'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                    'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                    'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                    'SIRSCritTemperature', 'SIRSCriteria2OrMore']

# possible values for case attributes
static_vals = {}
for attribute in static_cols:
    static_vals[attribute] = sepsis_data[attribute].unique().tolist()

# possible values for event attributes
dynamic_vals = {}
for attribute in dynamic_cols:
    dynamic_vals[attribute] = sepsis_data[attribute].unique().tolist()

# dict of orgs for each activity
orgs = {}
for activity in dynamic_vals["Activity"]:
    orgs[activity] = sepsis_data.loc[sepsis_data["Activity"] == activity, "org:group"].iloc[0]

# remove certain activity values that are obsolete or initialized otherwise
dynamic_vals["Activity"].remove("ER Registration")
dynamic_vals["Activity"].remove("Release A")
dynamic_vals["Activity"].remove("Release B")
dynamic_vals["Activity"].remove("Release C")
dynamic_vals["Activity"].remove("Release D")

def feature_generator(min_prefix_length: int, max_prefix_length: int):
    trace = [] # list of events
    static = {} # static case attributes

    # initialize case attributes with random values
    for attribute in static_cols:
        static[attribute] = random.choice(static_vals[attribute])

    for attribute in static_cols_bool:
    # initialize boolean case attributes with random values
        static[attribute] = random.choice(["True", "False"])

    timestamp = pd.Timestamp.now() # initialize timestamp
    timestamp += pd.Timedelta(days=random.randint(-100, 100), hours=random.randint(-24, 24), minutes=random.randint(-60, 60))
    
    static['Case ID'] = timestamp.timestamp() * 1000
    
    open_cases = random.randint(1, 101) # initialize number of open cases with random valueThe
    prefix_length = random.randint(min_prefix_length, max_prefix_length) # initialize length of trace
    crp = 0
    lactic_acid = 0
    leucocytes = 0
    # generate events and add to trace
    for i in range(1, prefix_length+1):
        event = static.copy() # initialize event with copy of case attributes
        event['event_nr'] = i # set sequential number of event in trace

        # if first event set activity to ER Registration
        if i == 1:
            event['Activity'] = "ER Registration"
        else:
            # set activity to random value
            event['Activity'] = random.choice(dynamic_vals['Activity'])


        # if activity CRP, get new value
        if event['Activity'] == "CRP":
            crp = random.choice(dynamic_vals["CRP"])
        # set CRP
        event["CRP"] = crp

        # if activity LacticAcid, get new value
        if event['Activity'] == "LacticAcid":
            lactic_acid = random.choice(dynamic_vals["LacticAcid"])
        # set LacticAcid
        event["LacticAcid"] = lactic_acid

        # if activity Leucocytes, get new value
        if event['Activity'] == "Leucocytes":
            leucocytes = random.choice(dynamic_vals["Leucocytes"])
        # set Leucocytes
        event["Leucocytes"] = leucocytes

        event['org:group'] = orgs[event['Activity']]  # set org:group value corresponding to Activity

        # set time values
        event["time:timestamp"] = timestamp
        event["timesincemidnight"] = timestamp.hour * 60 + timestamp.minute
        event["month"] = timestamp.month
        event["weekday"] = timestamp.weekday()
        event["hour"] = timestamp.hour

        # set open_cases and change value for next iteration
        event['open_cases'] = open_cases
        open_cases += random.randint(max(-open_cases, -25), 25)

        # randomly decide if next event follows instantly or not
        if bool(random.getrandbits(1)):
            # change timestamp for next iteration
            timestamp += pd.Timedelta(hours=random.randint(0, 24),
                                      minutes=random.randint(10, 60))

        trace.append(event) # append event to trace

    group = pd.DataFrame(trace) # convert trace to DataFrame

    # calculate and set timesincelastevent and timesincecasestart
    # algorithm adapted from preprocess_logs_sepsis_cases.extract_timestamp_features
    group = group.sort_values("time:timestamp", ascending=False, kind='mergesort')

    tmp = group["time:timestamp"] - group["time:timestamp"].shift(-1)
    tmp = tmp.fillna(pd.Timedelta(seconds=0))
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    tmp = group["time:timestamp"] - group["time:timestamp"].iloc[-1]
    tmp = tmp.fillna(pd.Timedelta(seconds=0))
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values("time:timestamp", ascending=True, kind='mergesort')

    # return finished trace as DataFrame
    return group

def feature_randomizer(trace, k: int):
    trace = list(trace.T.to_dict().values())
    k = min(k, len(trace))
    for i in range(k):
        random_index = random.randint(1, len(trace)-1)
        event_log = trace[random_index]
        for attribute in dynamic_cols:
            event_log[attribute] = random.choice(dynamic_vals[attribute])

    return pd.DataFrame(trace)

def synthesize(target_model, fixed_cls = 1,
               min_prefix_length = 5, max_prefix_length = 20,
               k_max = 10, k_min = 1, max_rejections = 10,
               min_prob = 0.1, max_prob = 1, target_prob = 0.8, max_iter = 1000):

    record = feature_generator(min_prefix_length, max_prefix_length)  # random record

    highest_prob = 0  # target model’s probability of fixed class
    rejections = 0  # consecutives rejections counter
    k = k_max
    i = max_iter
    while i > 0:
        probs = target_model.predict_proba(record)  # query target model
        current_prob = probs.flat[fixed_cls]

        # abort if minimum probability is not reached
        if current_prob < min_prob:
            return False

         # abort if maximum probability is  reached
        if current_prob > max_prob:
            return False

        if current_prob > highest_prob:
            print("new highest probability: %s"%(current_prob))
            if current_prob >= target_prob:
                return record
            # reset vars
            record_new = record
            highest_prob = current_prob
            rejections = 0
            # reward high probabilty with more tries
            i += 10 * current_prob * max_iter
            print("tries left: %s"%(i))
        else:
            rejections += 1
            if rejections > max_rejections:
                k = max(k_min, int(np.ceil(k / 2)))
                rejections = 0

            i -= 1

        record = feature_randomizer(record_new, k)

    return False

def synthesize_batch(container_array, target_number, target_model, fixed_cls = 1,
                    min_prob = 0.1,  max_prob = 1, target_prob = 0.8, max_iter = 1000):

    while len(container_array) < target_number:
        x = False

        while isinstance(x, pd.DataFrame) == False:  # repeat until synth finds record
            x = synthesize(target_model = target_model, fixed_cls = fixed_cls, min_prob = min_prob, max_prob = max_prob, target_prob = target_prob, max_iter = max_iter)

        container_array.append(x)

def remove_case_ids(group):
    static_vals["Case ID"].remove(group.iloc[0]["Case ID"])
    return group

def apply_key(group, key, value):
    group[key] = value
    return group

def apply_model_prob(group, model):
    proba = model.predict_proba(group)
    group["Class_0"] = proba[0][0]
    group["Class_1"] = proba[0][1]
    return group
    
def init_shadow_model_data(log_deviant, log_regular):
    dt_deviant = log_deviant.groupby("Case ID").apply(apply_key, key = "label", value = "deviant")
    dt_regular = log_regular.groupby("Case ID").apply(apply_key, key = "label", value = "regular")
    
    log = pd.concat([dt_deviant, dt_regular], ignore_index=True).sort_values("Case ID", ascending=True, kind="mergesort").reset_index(drop=True)
    
    return log
    
def init_attack_model_data(log, shadow_model):
    train, test = dataset_manager_sepsis.split_data_strict(log, 0.5, split="temporal")
    
    in_group = train.sort_values("Case ID", ascending=True, kind="mergesort").groupby("Case ID")
    attack_in_set = in_group.apply(apply_key, key = "label", value = "deviant")
    attack_in_set = attack_in_set.groupby("Case ID").apply(apply_model_prob, shadow_model)
    
    out_group = test.sort_values("Case ID", ascending=True, kind="mergesort").groupby("Case ID")
    attack_out_set = out_group.apply(apply_key, key = "label", value = "regular")
    attack_out_set = attack_out_set.groupby("Case ID").apply(apply_model_prob, shadow_model)
    
    attack_data_set = pd.concat([attack_in_set, attack_out_set], ignore_index=True).sort_values("Case ID", ascending=True, kind="mergesort").reset_index(drop=True)
    
    return attack_data_set

def train_attack_model(training_set, index : 1):
    with open("output/optimal_params_xgboost_synthetic_log_%s_single_index.pickle"%(index), "rb") as fin:
        args = pickle.load(fin)
        
    cls_encoder_args = {'case_id_col': "Case ID", 
                        'static_cat_cols': ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                                           'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                                           'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                                           'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                                           'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                                           'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                                           'SIRSCritTemperature', 'SIRSCriteria2OrMore'],
                        'static_num_cols': ['Age'], 
                        'dynamic_cat_cols': ["Activity", 'org:group'],
                        'dynamic_num_cols': ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", 
                                             "timesincemidnight", "timesincelastevent", "timesincecasestart", 
                                             "event_nr", "open_cases", 'Class_0', 'Class_1'], 
                        'fillna': True}
    
    cls = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators=500,
                            learning_rate= args['learning_rate'],
                            subsample=args['subsample'],
                            max_depth=int(args['max_depth']),
                            colsample_bytree=args['colsample_bytree'],
                            min_child_weight=int(args['min_child_weight']),
                            seed=22)
    
    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "index"]])
    
    pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
    
    train_y = dataset_manager_sepsis.get_label_numeric(training_set)
    
    pipeline.fit(training_set, train_y)
    
    pipeline_dump_filename = os.path.join("output", 'attack_model_%s.pkl'%(index))
    joblib.dump(pipeline, pipeline_dump_filename)
    
def test_attack_model(attack_test_set, attack_model, target_model, threshold):

    test_y_all = []
    preds_all = []
    probs_all = []

    attack_test_grouped = attack_test_set.groupby(dataset_manager_sepsis.case_id_col)

    for _, group in attack_test_grouped:

        test_y_all.extend(dataset_manager_sepsis.get_label_numeric(group))

        proba = target_model.predict_proba(group)
        group["Class_0"] = proba[0][0]
        group["Class_1"] = proba[0][1]
        
        prob = attack_model.predict_proba(group)[0][1]
        probs_all.append(prob)
        
        pred = 1 if prob > threshold else 0

        preds_all.append(pred)

    return (test_y_all, preds_all, probs_all)
    
def compute_precision_recall_curve(y_test, y_test_probs, n = 100):
    # Containers for true positive / false positive rates
    precision_scores = []
    recall_scores = []

    # Define probability thresholds to use, between 0 and 1
    probability_thresholds = np.linspace(0, 1, num=n)

    # Find true positive / false positive rate for each threshold
    for p in probability_thresholds:

        y_test_preds = []

        for prob in y_test_probs:
            if prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)

        precision = sklearn.metrics.precision_score(y_test, y_test_preds)
        recall = sklearn.metrics.recall_score(y_test, y_test_preds)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        
    return (precision_scores, recall_scores)
    
def plot_precision_recall_curve(recall_scores, precision_scores, label):
    fig, ax = plt.subplots(figsize=(6,6), dpi=200)
    
    ax.plot(recall_scores, precision_scores, label=label)
        
    baseline = len([num for num in recall_scores if num == 1]) / len(recall_scores)
    ax.plot([0, 1], [baseline, baseline], linestyle='--', label='Baseline')
        
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower right');

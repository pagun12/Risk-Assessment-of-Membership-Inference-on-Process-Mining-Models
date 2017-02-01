from sklearn.ensemble import RandomForestClassifier
from IndexBasedTransformer import IndexBasedTransformer
from sklearn.metrics import roc_auc_score
import pandas as pd
from HMMTransformer import HMMTransformer

datasets = {"bpic2015_%s_f%s"%(municipality, formula):"labeled_logs_csv/BPIC15_%s_f%s.csv"%(municipality, formula) for municipality in range(1,6) for formula in range(1,3)}
outfile = "results_hmm_bpic2015.csv"

prefix_lengths = list(range(2,21))

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"
dynamic_cols = ["monitoringResource", "question", "Resource"] # i.e. event attributes
static_cols_base = ["Responsible_actor", "SUMleges"] #+ list(dt_parts.columns) # i.e. case attributes that are known from the start
# maybe "caseProcedure" could also be used for labeling: normal or extended?
cat_cols_base = ["monitoringResource", "question", "Resource", "Activity", "Responsible_actor"]# + list(dt_parts.columns)

numeric_cols = ["SUMleges"]

train_ratio = 0.8
n_states = 6
n_iter = 30

def split_parts(group, parts_col="parts"):
    return pd.Series(group[parts_col].str.split(',').values[0], name='vals')

with open(outfile, 'w') as fout:
    for dataset_name, data_filepath in datasets.items():
        data = pd.read_csv(data_filepath, sep=";")

        data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
        data = data[data["caseStatus"] == "G"] # G is closed, O is open
        # replace infrequent questions with "other"
        data.loc[~data["question"].isin(data["question"].value_counts()[:4].index.tolist()), "question"] = "other"

        # switch labels (deviant/regular was set incorrectly before)
        data = data.set_value(col=label_col, index=(data[label_col] == pos_label), value="normal")
        data = data.set_value(col=label_col, index=(data[label_col] == neg_label), value=pos_label)
        data = data.set_value(col=label_col, index=(data[label_col] == "normal"), value=neg_label)

        # split the parts attribute to separate columns
        ser = data.groupby(level=0).apply(split_parts)
        dt_parts = pd.get_dummies(ser).groupby(level=0).apply(lambda group: group.max())
        data = pd.concat([data, dt_parts], axis=1)
        cat_cols = cat_cols_base + list(dt_parts.columns)
        static_cols = static_cols_base + list(dt_parts.columns)

        data[cat_cols] = data[cat_cols].fillna('missing')
        data = data.fillna(0)

        # split into train and test using temporal split
        grouped = data.groupby(case_id_col)
        start_timestamps = grouped[timestamp_col].min().reset_index()
        start_timestamps.sort_values(timestamp_col, ascending=1, inplace=True)
        train_ids = list(start_timestamps[case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[case_id_col].isin(train_ids)]
        test = data[~data[case_id_col].isin(train_ids)]


        for nr_events in prefix_lengths:
            hmm_transformer = HMMTransformer(n_states, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, pos_label, min_seq_length=2, max_seq_length=nr_events, random_state=22, n_iter=n_iter)
            hmm_transformer.fit(train)
            
            index_encoder = IndexBasedTransformer(nr_events, static_cols, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, activity_col)
            tmp = index_encoder.transform(train)
            
            dt_hmm = hmm_transformer.transform(train)
            hmm_merged = tmp.merge(dt_hmm, on=case_id_col)
            X = hmm_merged.drop([label_col, case_id_col], axis=1)
            y = hmm_merged[label_col]
            
            cls = RandomForestClassifier(n_estimators=500, random_state=22)
            cls.fit(X, y)


            test_encoded = index_encoder.transform(test)
            test_encoded = test_encoded.merge(hmm_transformer.transform(test), on=case_id_col)
            preds = cls.predict_proba(test_encoded.drop([label_col, case_id_col], axis=1))

            score = roc_auc_score([1 if label==pos_label else 0 for label in test_encoded[label_col]], preds[:,0])
            
            fout.write("%s;%s;%s\n"%(dataset_name, nr_events, score))
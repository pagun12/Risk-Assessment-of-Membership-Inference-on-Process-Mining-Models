from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.base import TransformerMixin

class HMMTransformer(TransformerMixin):
    
    def __init__(self, n_states, dynamic_cols, cat_cols, case_id_col, timestamp_col, label_col, pos_label, min_seq_length=2, max_seq_length=None, random_state=None, n_iter=10):
        
        self.pos_hmms = None
        self.neg_hmms = None
        self.pos_encoders = None
        self.neg_encoders = None
        
        self.n_states = n_states
        self.dynamic_cols = dynamic_cols
        self.cat_cols = cat_cols
        self.case_id_col = case_id_col
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.pos_label = pos_label
        self.min_seq_length = min_seq_length
        
        if max_seq_length is None:
            self.max_seq_length = 100000000
        else:
            self.max_seq_length = max_seq_length
            
        self.columns = None
        
    
    def fit(self, X, y=None):
        
        self.pos_hmms, self.pos_encoders = self._train_hmms(X[X[self.label_col] == self.pos_label])
        self.neg_hmms, self.neg_encoders = self._train_hmms(X[X[self.label_col] != self.pos_label])
        
        return self
    
    
    def transform(self, X):
        scores = X.groupby(self.case_id_col).apply(self._calculate_scores)
        dt_scores = pd.DataFrame.from_records(list(scores.values), columns=["hmm_%s"%col for col in self.dynamic_cols])
        dt_scores[self.case_id_col] = scores.index
        
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_scores.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_scores.columns]
            for col in missing_cols:
                dt_scores[col] = 0
            dt_scores = dt_scores[self.columns]
            
        return dt_scores
    
    
    def _train_hmms(self, X):

        grouped = X.groupby(self.case_id_col)
        hmms = {}
        encoders = {}

        for col in self.dynamic_cols:
            tmp_dt_hmm = []
            for name, group in grouped:
                if len(group) >= self.min_seq_length:
                    seq = [val for val in group.sort_values(self.timestamp_col, ascending=1)[col]]
                    if self.max_seq_length is not None:
                        seq = seq[:self.max_seq_length]
                    tmp_dt_hmm.extend(seq)
            if col in self.cat_cols:
                hmms[col] = hmm.MultinomialHMM(n_components=self.n_states)
                #encoders[col] = LabelEncoder()
                #tmp_dt_hmm = encoders[col].fit_transform(tmp_dt_hmm)
                encoders[col] = {label:idx for idx, label in enumerate(set(tmp_dt_hmm))}
                tmp_dt_hmm = [encoders[col][label] for label in tmp_dt_hmm]
                
            else:
                hmms[col] = hmm.GaussianHMM(n_components=self.n_states)
                
            hmms[col] = hmms[col].fit(np.atleast_2d(tmp_dt_hmm).T, [min(val, self.max_seq_length) for val in grouped.size() if val >= self.min_seq_length])
            
            # check for buggy transition matrix
            rowsums = hmms[col].transmat_.sum(axis=1)
            if rowsums.sum() < len(rowsums):
                for problem_row_idx in np.where(rowsums < 1)[0]:
                    hmms[col].transmat_[problem_row_idx] = 1.0 / hmms[col].transmat_.shape[1]
                

        return (hmms, encoders)
    
    
    def _calculate_scores(self, group):
        
        if len(group) < self.min_seq_length:
            return tuple([0] * len(self.dynamic_cols))
        
        scores = []
        
        for col in self.dynamic_cols:
            tmp_dt_hmm = [val for val in group.sort_values(self.timestamp_col, ascending=1)[col]]
            if self.max_seq_length is not None:
                tmp_dt_hmm = tmp_dt_hmm[:self.max_seq_length]
            
            if col in self.cat_cols:
                #tmp_dt_hmm_pos = self.pos_encoders[col].transform(tmp_dt_hmm)
                #tmp_dt_hmm_neg = self.neg_encoders[col].transform(tmp_dt_hmm)
                
                tmp_dt_hmm_pos = [self.pos_encoders[col][label] for label in tmp_dt_hmm if label in self.pos_encoders[col]]
                tmp_dt_hmm_neg = [self.neg_encoders[col][label] for label in tmp_dt_hmm if label in self.neg_encoders[col]]
            
                pos_score = 0
                neg_score = 0
                if len(tmp_dt_hmm_pos) >= self.min_seq_length:
                    pos_score = self.pos_hmms[col].score(np.atleast_2d(tmp_dt_hmm_pos).T)
                if len(tmp_dt_hmm_neg) >= self.min_seq_length:   
                    neg_score = self.neg_hmms[col].score(np.atleast_2d(tmp_dt_hmm_neg).T)
                
            else:
                pos_score = self.pos_hmms[col].score(np.atleast_2d(tmp_dt_hmm).T)
                neg_score = self.neg_hmms[col].score(np.atleast_2d(tmp_dt_hmm).T)
            
            scores.append(pos_score - neg_score)
        
        return tuple(scores)
    
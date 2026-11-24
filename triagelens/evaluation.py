"""Evaluation of frozen policies; never tune on evaluation or uploaded queries."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def policy_metrics(scores,threshold,correct,is_oos,prevalence=.18):
    scores=np.asarray(scores,float); correct=np.asarray(correct,bool); is_oos=np.asarray(is_oos,bool)
    if len(scores)==0 or not(len(scores)==len(correct)==len(is_oos)) or not np.isfinite(scores).all():
        raise ValueError('Invalid evaluation vectors.')
    observed=is_oos.mean()
    if not 0<observed<1 or not 0<prevalence<1:
        raise ValueError('Weighted policy evaluation requires known and OOS examples.')
    if correct[is_oos].any():
        raise ValueError('OOS predictions must count as incorrect.')
    weights=np.where(is_oos,prevalence/observed,(1-prevalence)/(1-observed))
    accepted=scores>=threshold; wrong=accepted & ~correct
    return {'Coverage':float(np.average(accepted,weights=weights)),
            'Selective risk':float(weights[wrong].sum()/weights[accepted].sum()) if accepted.any() else np.nan,
            'OOS leakage':float(weights[accepted & is_oos].sum()/weights[is_oos].sum()),
            'Expected cost':float(np.average(np.where(accepted,np.where(correct,0,10),1),weights=weights)),
            'Automated (count)':int(accepted.sum()),'Wrong automated (count)':int(wrong.sum())}

def classification_metrics(y_true,y_pred,classes):
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=classes,average='macro',zero_division=0)
    return {'Accuracy':accuracy_score(y_true,y_pred),'Macro precision':precision,'Macro recall':recall,'Macro F1':f1},confusion_matrix(y_true,y_pred,labels=classes)

def evaluate_dataset(predictor,data,progress=None):
    from .inference import predict
    rows=[]
    records=data['test']+data['oos_test']
    for i,(text,label) in enumerate(records):
        r=predict(predictor,text)
        rows.append({'text':text,'true_intent':label,'predicted_intent':r['predicted_intent'],
                     'secondary_intent':r['secondary_intent'],'confidence':r['confidence'],'trust_score':r['trust_score']})
        if progress and (i%25==0 or i==len(records)-1):progress((i+1)/len(records))
    frame=pd.DataFrame(rows); oos=frame.true_intent.eq('oos').to_numpy()
    correct=(~oos)&frame.true_intent.eq(frame.predicted_intent).to_numpy()
    policies=pd.DataFrame([dict(Method=name,**policy_metrics(frame[col],threshold,correct,oos)) for name,col,threshold in [
        ('Confidence only','confidence',predictor.confidence_threshold),('Seven-signal trust','trust_score',predictor.trust_threshold)]])
    metrics=[]; matrices={}
    for name,col in [('TF-IDF + LR','predicted_intent'),('LSA + LR','secondary_intent')]:
        m,cm=classification_metrics(frame.loc[~oos,'true_intent'],frame.loc[~oos,col],predictor.primary.classes_)
        metrics.append(dict(Model=name,**m));matrices[name]=cm
    return frame,policies,pd.DataFrame(metrics),matrices

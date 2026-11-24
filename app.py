"""Run with: streamlit run app.py"""
import json
import logging
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from triagelens.config import ROOT,ARTIFACT_ROOT,SAMPLES,SIGNAL_HELP,missing_artifacts
from triagelens.data import parse_upload,load_dataset
from triagelens.evidence import tables,outputs
from triagelens.models import load_predictor,ArtifactError
from triagelens.inference import predict
from triagelens.preprocessing import transform,feature_table
from triagelens.visualization import bars,workflow
from triagelens.evaluation import evaluate_dataset

st.set_page_config(page_title='TriageLens · Research studio',page_icon='◈',layout='wide',initial_sidebar_state='auto')
st.markdown('''<style>
.block-container{padding-top:2.5rem;max-width:1450px}h1{letter-spacing:-.045em;font-weight:750}h2{letter-spacing:-.025em}
[data-testid="stMetric"]{background:linear-gradient(130deg,#17243b,#111a2c);border:1px solid #293650;border-radius:14px;padding:20px}
[data-testid="stMetricLabel"]{color:#94a3b8}[data-testid="stSidebar"]{border-right:1px solid #293650}
.hero{padding:30px 32px;border:1px solid #2b4158;background:linear-gradient(115deg,#122f38,#16213c);border-radius:18px;margin:12px 0 24px}
.eyebrow{color:#5eead4;font-size:.75rem;letter-spacing:.18em;font-weight:700}.hero p{max-width:850px;color:#cbd5e1;line-height:1.7}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Validating notebook artifacts…')
def cached_predictor(root,signature):
    return load_predictor(root)

def get_predictor():
    paths=list((ARTIFACT_ROOT/'triagelens_models').glob('*.joblib'))+[ARTIFACT_ROOT/'triagelens_results/experiment_config.json']
    signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(paths) if p.exists())
    return cached_predictor(str(ARTIFACT_ROOT),signature)

def show_missing():
    st.warning('Live inference is unavailable: the trained notebook artifacts have not been supplied.')
    st.write('Export from the original notebook kernel using the procedure on **Artifacts & setup**. Recorded research evidence remains available.')
    with st.expander('Required files',expanded=False):
        st.code('\n'.join(missing_artifacts(ARTIFACT_ROOT)))

def show_table(cell,description):
    st.caption(description+f' · Notebook cell {cell} (zero-based) · displayed precision')
    found=tables(cell)
    if not found:
        st.info('No table was preserved in this notebook output.')
    for filename,frame in found:
        st.dataframe(frame,use_container_width=True)
        st.download_button('Download displayed table',frame.to_csv().encode(),file_name=filename.replace('.html','.csv'),key='dl_'+filename)

def header(kicker,title,subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>',unsafe_allow_html=True)
    st.title(title)
    st.write(subtitle)

with st.sidebar:
    st.markdown('## ◈ TriageLens')
    st.caption('RESEARCH DEMONSTRATION STUDIO')
    page=st.radio('Workspace',['Overview','Query lab','Representations','Results & comparison','Methodology','Artifacts & setup'],label_visibility='collapsed')
    st.divider()
    ready=not missing_artifacts(ARTIFACT_ROOT)
    if ready:
        st.success('Artifact files present')
    else:
        st.warning('Awaiting trained artifacts')
    if (ARTIFACT_ROOT/'triagelens_results/reproduction_run.json').exists():
        st.caption('Models retrained from your uploaded dataset using the original notebook pipeline.')
    st.caption('CLINC150 · 150 known intents\n\nTF-IDF + LSA · 7 reliability signals')
    st.caption('AUTOMATE accepts an intent label. It executes no external action.')

if page=='Overview':
    header('SELECTIVE CLASSIFICATION','Better decisions begin with knowing when to ask.',
           'Confidence-aware human-in-the-loop intent classification')
    st.markdown('''<div class="hero"><span class="eyebrow">THE RESEARCH QUESTION</span><h2>Which predictions deserve automation?</h2><p>TriageLens predicts what a user wants, then combines seven reliability signals to decide whether to accept the intent or send it for human review. The objective is to reduce decision cost while balancing automation coverage, errors and unfamiliar requests.</p></div>''',unsafe_allow_html=True)
    cols=st.columns(4)
    for col,label,value in zip(cols,['Known intent classes','Reliability signals','Recorded validation accuracy','Recorded test cost savings'],['150','7','88.33%','8.75%']):
        col.metric(label,value)
    st.caption('Counts and metrics above are persisted notebook evidence. Accuracy is primary TF-IDF LR on 3,000 known validation queries; cost savings compare frozen policies on final test, weighted to 18% OOS with wrong automation:review cost 10:1. These are different evaluation criteria.')
    st.subheader('Explore the decision pipeline')
    st.plotly_chart(workflow(),use_container_width=True,config={'displayModeBar':False})
    st.caption('Hover over each stage for its role. The secondary LSA path and both classifiers contribute reliability evidence; the primary model supplies the intent.')
    a,b=st.columns([1.15,1])
    with a:
        st.subheader('Language in, informed decisions out')
        st.write('The notebook records 23,700 CLINC150 queries across six splits: 15,000 train, 3,000 validation, 4,500 test, and 100/100/1,000 out-of-scope train/validation/test examples. OOS training data are not used by this pipeline.')
        st.write('Two logistic-regression models read lexical and latent semantic representations. A separate weighted logistic-regression model estimates whether the primary prediction is correct.')
    with b:
        st.subheader('Demonstration readiness')
        if ready:st.success('Model files are available. Query lab validates them before inference.')
        else:show_missing()
        st.info('The supplied project processes text. Signal graphs, GAF, recurrence plots, spectrograms and neural-network learning curves do not apply.')

elif page in ('Query lab','Representations'):
    header('LIVE RESEARCH WORKSPACE',page,'Inspect an actual query, its fitted representations and the frozen research decision.')
    source=st.radio('Input source',['Notebook demonstration query','Type a query','Upload queries','Local dataset sample'],horizontal=True)
    query=None
    if source=='Notebook demonstration query':
        query=st.selectbox('Example from notebook cells 41 / 44',SAMPLES)
        st.caption('Literal demonstration queries from the notebook; not claimed to be held-out dataset records. No saved prediction is used here.')
    elif source=='Type a query':
        query=st.text_area('Natural-language query',max_chars=10000,placeholder='Describe what you want to do…')
    elif source=='Upload queries':
        file=st.file_uploader('UTF-8 TXT (one query), CSV (text column), JSON (list of strings or {text: …})',type=['txt','csv','json'])
        if file:
            try:
                queries=parse_upload(file.getvalue(),file.name)
                index=st.selectbox('Select a query row',range(len(queries)),format_func=lambda i:f'{i+1}: {queries[i][:100]}')
                query=queries[index]
                st.caption(f'{len(queries)} validated queries · 200-row / 2-MB upload limit')
            except ValueError as exc:st.error(str(exc))
    else:
        dataset_path=ARTIFACT_ROOT/'data_full.json'
        if not dataset_path.exists():st.info('Place the original data_full.json in the configured artifact directory to choose genuine dataset samples.')
        else:
            try:
                data=load_dataset(dataset_path)
                split=st.selectbox('Dataset split',list(data))
                idx=st.number_input('Record index (zero-based)',min_value=0,max_value=len(data[split])-1,value=0)
                query,label=data[split][idx]
                st.caption(f'Ground truth: {label} · {split} · record {idx}')
            except (ValueError,OSError,json.JSONDecodeError):st.error('Dataset cannot be read or does not match the original CLINC150 split schema.')
    if query is not None:
        st.subheader('Original input')
        st.text(query)
        a,b,c=st.columns(3);a.metric('Characters',len(query));b.metric('Whitespace-separated words',len(query.split()));c.metric('Input type','Text')
    if not ready:
        show_missing()
        st.button('Analyze query',disabled=True,type='primary')
    else:
        try:p=get_predictor()
        except ArtifactError as exc:
            st.error(str(exc));st.stop()
        if st.button('Analyze query',type='primary',disabled=not query or not query.strip()):
            try:
                with st.spinner('Transforming query and running frozen models…'):
                    result=predict(p,query)
                st.session_state['analysis']={'query':query,'root':str(ARTIFACT_ROOT),'result':result}
            except ValueError as exc:st.error(str(exc))
            except Exception:
                logging.exception('Inference failed')
                st.error('Inference could not complete. Check artifact compatibility in Artifacts & setup.')
        saved=st.session_state.get('analysis')
        if saved and saved['query']==query and saved['root']==str(ARTIFACT_ROOT):
            r=saved['result']
            x,raw,z=transform(p,query)
            if r['zero_vocabulary']:st.warning('No query terms occur in the fitted vocabulary. The models still return their learned outputs; this input provides no lexical evidence and the trust policy is not a safety guarantee.')
            if page=='Query lab':
                st.divider();st.subheader(r['predicted_intent'].replace('_',' ').title())
                st.code(r['predicted_intent'])
                c1,c2,c3=st.columns(3)
                c1.metric('Primary confidence',f"{r['confidence']:.2%}")
                c2.metric('Learned trust',f"{r['trust_score']:.2%}")
                c3.metric('Trust threshold margin',f"{r['trust_score']-p.trust_threshold:+.4f}")
                a,b=st.columns(2)
                with a:
                    st.write(f'**Confidence-only · threshold {p.confidence_threshold:.6f}**')
                    st.info(r['confidence_decision'])
                with b:
                    st.write(f'**Seven-signal policy · threshold {p.trust_threshold:.6f}**')
                    if r['triagelens_decision']=='AUTOMATE':
                        st.success(r['triagelens_decision'])
                    else:
                        st.warning(r['triagelens_decision'])
                if abs(r['trust_score']-p.trust_threshold)<.02:st.warning('Decision is within 0.02 of the frozen trust threshold.')
                st.caption(f"Measured inference-call time on this machine: {r['elapsed_ms']:.1f} ms (single call; excludes loading and charting; not a research latency benchmark).")
                prob=pd.DataFrame({'Intent':r['classes'],'TF-IDF + LR':r['primary_probabilities'],'LSA + LR':r['secondary_probabilities']})
                top=prob.nlargest(10,'TF-IDF + LR')
                st.plotly_chart(px.bar(top.melt(id_vars='Intent',var_name='Model',value_name='Probability'),x='Intent',y='Probability',color='Model',barmode='group',color_discrete_sequence=['#2dd4bf','#818cf8'],title='Both actual classifier distributions · primary top 10 intents'),use_container_width=True)
                st.caption(f"Secondary classifier intent: {r['secondary_intent']}. Probabilities cover known intents only and are not a probability that the query is in scope.")
                st.dataframe(prob.sort_values('TF-IDF + LR',ascending=False),hide_index=True,use_container_width=True)
                st.subheader('Seven reliability signals')
                signals=pd.DataFrame([{'Signal':k,'Value':v,'Meaning':SIGNAL_HELP[k]} for k,v in r['signals'].items()])
                st.dataframe(signals,hide_index=True,use_container_width=True)
                st.caption('Signals explain the inputs to the trust decision. They do not establish causal reasons for a classification. Trust is a statistical correctness estimate, not certainty. Review does not necessarily imply OOS.')
                st.download_button('Download actual inference result',json.dumps(r,indent=2),file_name='triagelens_prediction.json',mime='application/json')
            else:
                st.subheader('1 · Fitted lexical preprocessing')
                st.write('The training-fitted TfidfVectorizer lowercases text, applies its default word-token pattern, builds unigrams and bigrams, applies 1 + log(term frequency), multiplies by fitted IDF, and L2-normalizes the vector. No extra cleaning or fitting is performed.')
                st.write('Analyzer output:',p.tfidf.build_analyzer()(query))
                st.caption(f'Model-ready sparse shape: {x.shape} · nonzero features: {x.nnz} · L2 norm: {float((x.multiply(x).sum())**.5):.6f}')
                f=feature_table(p,x)
                if f.empty:st.info('No vocabulary matches.')
                else:
                    st.plotly_chart(bars(f.head(25),'Term / n-gram','TF-IDF weight','Largest active fitted TF-IDF features'),use_container_width=True)
                    st.dataframe(f,hide_index=True,use_container_width=True)
                st.subheader('2 · Latent semantic representation')
                st.write('The original training-fitted TruncatedSVD projects the lexical vector to 300 dimensions. Row L2 normalization makes dot products suitable for cosine similarity. Component indices are latent axes, not words or time steps.')
                frame=pd.DataFrame({'Component':range(raw.shape[1]),'Before normalization':raw[0],'Model-ready LSA':z[0]})
                st.plotly_chart(px.line(frame,x='Component',y=['Before normalization','Model-ready LSA'],title='Actual LSA values before and after normalization'),use_container_width=True)
                st.caption(f'Final model input: {z.shape} · norm: {float((z*z).sum()**.5):.6f} · fitted explained variance: {p.lsa.explained_variance_ratio_.sum():.4f}')
                st.dataframe(frame,hide_index=True,use_container_width=True)
                st.info('No 1D-to-image encoding is part of this research. Both representations above are computed by the exported training transforms.')

elif page=='Results & comparison':
    header('EVIDENCE & EVALUATION','Results with a traceable source.','Explore recorded research evidence or recompute held-out results with the original artifacts and dataset.')
    mode=st.radio('Evidence source',['Recorded notebook outputs','Recompute with local artifacts'],horizontal=True)
    if mode=='Recorded notebook outputs':
        st.info('These are persisted outputs from the uploaded notebook, at their original display precision. They have not been reproduced here. Full-precision exported files, when supplied, are available under Artifacts & setup.')
        section=st.selectbox('Experiment',['Classifier validation','Frozen final-test policies','Bootstrap uncertainty','Feature ablation','Calibration','Cost sensitivity','Risk–coverage samples','Full output audit'])
        if section=='Classifier validation':
            df=pd.DataFrame({'Model':['TF-IDF + LR','LSA + LR'],'Accuracy':[.8833,.8307],'Macro F1':[.8826,.8280]})
            st.plotly_chart(px.bar(df.melt(id_vars='Model',var_name='Metric',value_name='Score'),x='Model',y='Score',color='Metric',barmode='group',color_discrete_sequence=['#2dd4bf','#818cf8']),use_container_width=True)
            st.dataframe(df,hide_index=True,use_container_width=True)
            st.caption('Cell 20; known-intent validation only, 3,000 queries. Primary TF-IDF LR has higher recorded accuracy. This ranking concerns classification, not selective-policy cost.')
        elif section=='Frozen final-test policies':
            show_table(32,'Final test: 4,500 known + 1,000 OOS; weighted to 18% OOS; frozen policy thresholds; wrong automation:review = 10:1')
            st.write('Recorded expected cost: confidence-only **0.7276**, seven-signal trust **0.6639**. The seven-signal policy has the lower recorded cost, with **8.75%** relative savings. Costs are illustrative decision units per incoming query.')
        elif section=='Bootstrap uncertainty':show_table(33,'3,000 paired stratified final-test bootstrap repetitions; 95% percentile confidence intervals. Positive cost savings favor trust; positive risk/leakage differences mean increases.')
        elif section=='Feature ablation':show_table(34,'Exploratory policy validation only; weighted correctness AUROC, not intent-classification AUROC. Configurations refit on trust training only.')
        elif section=='Calibration':
            show_table(36,'Validation and final test reported separately; weighted Brier and ECE (10 equal-width bins). Lower is better.')
            show_table(35,'Policy-validation comparison and trust reliability bins, recorded rounded values')
        elif section=='Cost sensitivity':show_table(29,'Validation-optimized policies at cost ratios 5:1, 10:1, 20:1. Not final test performance.')
        elif section=='Risk–coverage samples':show_table(27,'Policy-validation weighted comparisons at achievable coverage not exceeding target; these are sampled table points, not the full curve.')
        else:
            cell=st.selectbox('Source cell',range(48))
            for o in outputs():
                if o['cell']==cell:st.text(o['text'])
            show_table(cell,'Raw preserved output tables; may be truncated by the original notebook display')
    else:
        st.write('Uses all 5,500 final-test queries and the frozen original models. No fitting or threshold selection occurs. This CPU operation may take several minutes.')
        dataset_path=ARTIFACT_ROOT/'data_full.json'
        can_run=ready and dataset_path.exists()
        if not can_run:st.warning('Requires all exported models, experiment_config.json and the original data_full.json.')
        if st.button('Evaluate frozen final test',disabled=not can_run,type='primary'):
            try:
                p=get_predictor();data=load_dataset(dataset_path)
                progress=st.progress(0.0,text='Computing held-out predictions…')
                with st.spinner('Evaluating genuine test records…'):
                    st.session_state['evaluation']=evaluate_dataset(p,data,progress.progress)
                progress.empty()
            except (ValueError,OSError,json.JSONDecodeError) as exc:st.error(str(exc))
            except Exception:
                logging.exception('Evaluation failed');st.error('Evaluation could not complete. Check original artifacts and dataset compatibility.')
        if 'evaluation' in st.session_state:
            frame,policies,metrics,matrices=st.session_state['evaluation']
            st.caption('Recomputed in this app · known-only classifier metrics; policy metrics weighted to 18% OOS, error/review cost 10:1.')
            st.dataframe(policies,hide_index=True,use_container_width=True)
            st.plotly_chart(bars(policies,'Method','Expected cost','Frozen policy cost · recomputed'),use_container_width=True)
            st.dataframe(metrics,hide_index=True,use_container_width=True)
            model=st.selectbox('Confusion matrix model',list(matrices))
            p=get_predictor();labels=p.primary.classes_
            st.plotly_chart(px.imshow(matrices[model],x=labels,y=labels,labels={'x':'Predicted intent','y':'Actual intent','color':'Count'},title='Known-intent final-test confusion matrix (raw counts)',color_continuous_scale='Teal'),use_container_width=True)
            st.download_button('Download recomputed test records',frame.to_csv(index=False),file_name='recomputed_test_predictions.csv')

elif page=='Methodology':
    header('METHODS & INTERPRETATION','From language to selective automation.','The workflow implemented in the supplied notebook, including its validation boundaries.')
    st.plotly_chart(workflow(),use_container_width=True)
    for title,body in [
        ('1 · Fit on known training queries','Fit TF-IDF, 300-component SVD and both intent classifiers on 15,000 known training examples. LSA vectors are L2-normalized. The lexical and semantic views supply complementary evidence.'),
        ('2 · Extract reliability evidence','Calculate the seven signals on 3,000 known validation and 100 OOS validation examples. Compare queries to the training references only.'),
        ('3 · Separate learning from policy selection','Stratify validation into 1,550 trust-training and 1,550 policy-validation records using OOS / known-correct / known-incorrect groups, seed 42. Fit the weighted scaler and correctness classifier on trust training only.'),
        ('4 · Choose policy thresholds, then freeze','On policy validation only, minimize weighted expected cost = coverage × selective risk × 10 + (1 − coverage) × 1. Assume 18% OOS. Include review-everything as an option. Freeze both thresholds before final testing.'),
        ('5 · Evaluate once on held-out test','Evaluate 4,500 known and 1,000 OOS queries. All automated OOS predictions count as wrong. Compare frozen confidence and trust policies. Bootstrap uses paired stratified resampling; ablation and sensitivity remain validation analyses.'),
        ('6 · Demonstrate without refitting','Load original artifacts and apply the same transforms and seven-signal policy to user text. The trust score estimates correctness; it does not change the primary intent label.')]:
        with st.expander(title,expanded=True):st.write(body)
    st.subheader('Reading the outcomes')
    st.write('Coverage is the weighted fraction accepted. Selective risk is the weighted error fraction among accepted queries. OOS leakage is the fraction of OOS queries accepted. Accuracy and macro-F1 measure known-intent classification. Correctness AUROC measures ranking of correct versus incorrect/OOS predictions, not multi-class intent AUROC.')
    st.subheader('Limits and defensible next steps')
    st.write('A trust score is not certainty. An unfamiliar request may receive a confident known-intent label. Assumed prevalence and costs may differ from deployment; calibration is scenario-dependent. The provided attachment cannot establish live numerical parity without its model artifacts. No external action is executed. No neural-network attribution, training curves or invented image encodings are added.')
    st.write('Future work: test domain and prevalence shift on a new untouched evaluation set, inspect group-specific errors and calibration, and measure end-to-end deployment latency. Keep future policy tuning separate from final evaluation.')
    st.download_button('Download complete technical analysis',(ROOT/'ANALYSIS.md').read_text(),file_name='ANALYSIS.md')

else:
    header('REPRODUCIBILITY','Connect the original research artifacts.','The notebook is preserved unchanged. Export the trained objects from your existing notebook kernel to enable inference.')
    st.code(f'Artifact directory: {ARTIFACT_ROOT}')
    missing=missing_artifacts(ARTIFACT_ROOT)
    if missing:st.dataframe(pd.DataFrame({'Missing required file':missing}),hide_index=True,use_container_width=True)
    else:
        try:
            p=get_predictor();st.success('All model artifacts passed compatibility checks.')
            st.json({'confidence_threshold':p.confidence_threshold,'trust_threshold':p.trust_threshold,'reference_shape':list(p.training_tfidf.shape),'lsa_shape':list(p.training_lsa.shape)})
        except ArtifactError as exc:st.error(str(exc))
    st.subheader('Export from the completed notebook')
    st.write('In the original notebook environment, after executing through cell 41 (zero-based), run the following new cell. The helper exports the actual fitted objects, exact thresholds, result tables, training texts and full-precision parity fixtures. It also verifies the serialized inference against the live notebook function.')
    st.code('''import sys
sys.path.insert(0, r"/absolute/path/to/triagelens")
from scripts.export_from_notebook import export_from_namespace
export_from_namespace(globals(), r"/absolute/path/to/triagelens")''',language='python')
    st.write('Alternatively, copy the triagelens_models and triagelens_results directories created by original notebook cells 37–38. Exporting with the helper additionally enables exact parity verification. Only load trusted local joblib artifacts, since Python deserialization can execute code.')
    st.subheader('Setup and validation commands')
    st.code('''python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_artifacts.py
streamlit run app.py''',language='bash')
    st.caption('Use Python 3.11 or 3.12. Pinned scientific packages match the notebook record. PyTorch and transformer libraries are not required by its inference pipeline. Set TRIAGELENS_PROJECT_DIR to use artifacts from another local directory; restart the app after changing it.')
    package=ROOT.parent/'TriageLens.zip'
    if package.is_file():
        with st.expander('Download the ready-to-run application'):
            st.download_button('Download TriageLens ZIP',package.read_bytes(),file_name='TriageLens.zip',mime='application/zip')
            st.caption('Extract the ZIP on Windows and double-click start_windows.bat. Initial setup needs Python 3.11/3.12 and internet for dependencies.')
    st.subheader('Optional original dataset')
    st.write('Copy the original data_full.json beside app.py or into TRIAGELENS_PROJECT_DIR. The app then exposes genuine split samples and held-out recomputation. It does not download a potentially different dataset or silently retrain models.')
    results=ARTIFACT_ROOT/'triagelens_results'
    for path in sorted(results.glob('*.csv')) if results.exists() else []:
        if path.stat().st_size>10_000_000:continue
        with st.expander('Local exported file · '+path.name):
            try:
                st.caption('Original local export; not recomputed by this app session. See the README for each table’s evaluation split.')
                st.dataframe(pd.read_csv(path),use_container_width=True)
                st.download_button('Download original export',path.read_bytes(),file_name=path.name,key='export_'+path.name)
            except (OSError,ValueError):st.error('Unable to read this result table.')
    run_record=results/'reproduction_run.json'
    if run_record.exists():
        st.subheader('Verified reproduction from your uploaded dataset')
        st.json(json.loads(run_record.read_text()))
    st.subheader('Source provenance')
    st.json(json.loads((ROOT/'evidence/provenance.json').read_text()))
    st.caption('The packaged evidence is separate from live inference. Displayed rounded thresholds are never used as fallback inference settings.')

st.divider()
st.caption('TriageLens · CLINC150 research studio · Evidence first, frozen policies, real local inference when artifacts are available')

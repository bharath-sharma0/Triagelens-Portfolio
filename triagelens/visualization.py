import plotly.express as px
import plotly.graph_objects as go

def bars(frame,x,y,title):
    figure=px.bar(frame,x=x,y=y,title=title,color_discrete_sequence=['#2dd4bf'])
    figure.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=55,b=30,l=20,r=20))
    return figure

def workflow():
    labels=['Query','TF-IDF','LSA + L2','Two classifiers','7 signals','Trust model','Accept / review']
    fig=go.Figure()
    for i,label in enumerate(labels):
        fig.add_trace(go.Scatter(x=[i],y=[0],mode='markers+text',text=[label],textposition='top center',marker=dict(size=22,color='#2dd4bf'),hovertext=[[
            'Raw natural-language input; no extra cleaning.', 'Fitted training vocabulary, sublinear TF and L2 normalization.',
            'Training-fitted 300-component SVD; normalize each resulting vector.', 'TF-IDF primary and LSA secondary logistic regression.',
            'Probabilities, training-neighbor similarities, consensus and model agreement.',
            'Frozen weighted StandardScaler + logistic regression.', 'Apply full-precision frozen threshold; no external action.'][i]],hoverinfo='text',showlegend=False))
        if i:fig.add_annotation(x=i-.12,y=0,ax=i-1+.12,ay=0,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowcolor='#64748b',arrowhead=2)
    fig.update_layout(height=175,margin=dict(l=50,r=65,t=30,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(visible=False,range=[-.4,6.5]),yaxis=dict(visible=False,range=[-1,1]),font=dict(color='#cbd5e1'))
    return fig

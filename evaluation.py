import streamlit as st
import plotly.figure_factory as ff
import pandas as pd

def render_evaluation_page(metrics):
    st.title("Model Evaluation")
    st.markdown("<p style='color: #64748b;'>Evaluate the performance of the NLP sentiment classification model.</p>", unsafe_allow_html=True)
    
    if "error" in metrics:
        st.error(f"Could not load evaluation metrics. Error: {metrics['error']}")
        return
        
    st.info("Performance calculated on a 20% unseen test split.")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    col_cm, col_cr = st.columns([1.2, 1])
    
    with col_cm:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        
        cm = metrics.get('confusion_matrix', [])
        classes = metrics.get('classes', [])
        
        if len(cm) > 0 and len(classes) > 0:
            fig = ff.create_annotated_heatmap(
                z=cm, 
                x=list(classes), 
                y=list(classes),
                colorscale='Blues',
                showscale=False
            )
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                margin=dict(t=40, b=40, l=40, r=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#0f172a")
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col_cr:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; height: 100%;'>", unsafe_allow_html=True)
        st.subheader("Classification Report")
        
        report = metrics.get('classification_report', {})
        if report:
            report_data = []
            for class_name in classes:
                if class_name in report:
                    row = report[class_name]
                    report_data.append({
                        "Class": class_name,
                        "Precision": f"{row['precision']:.2f}",
                        "Recall": f"{row['recall']:.2f}",
                        "F1": f"{row['f1-score']:.2f}",
                        "Support": int(row['support'])
                    })
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        st.markdown("### Model Pipeline")
        st.markdown("""
        **1.** News Text  
        **↓**  
        **2.** Text Preprocessing (Clean, Stopwords, Lemmatize)  
        **↓**  
        **3.** TF-IDF Vectorization  
        **↓**  
        **4.** Logistic Regression  
        **↓**  
        **5.** Sentiment Prediction
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="NLP Preprocessing Demo", layout="wide")
st.title("🧠 NLP Preprocessing Suite")

tab1, tab2, tab3 = st.tabs(["🔍 Preprocessing", "📊 Stem vs Lemma Comparison", "📈 TF-IDF Embeddings"])

# --- TAB 1: Preprocessing ---
with tab1:
    st.subheader("Text Preprocessing")

    user_input = st.text_area("Enter your text:", height=150)

    if st.button("Run NLP Preprocessing"):
        with st.spinner("Processing..."):
            res = requests.post(f"{API_URL}/preprocess", json={"text": user_input})
        
        if res.status_code == 200:
            data = res.json()
            st.success("Preprocessing complete!")

            # Display Tokens
            with st.expander("🧩 Tokens"):
                st.markdown("#### Token List")
                st.code(", ".join(data["tokens"]), language="markdown")

            # Display Lemmas and Stems in a table
            with st.expander("📚 Lemmas & 🌱 Stems Table"):
                table_data = pd.DataFrame({
                    "Token": data["tokens"],
                    "Lemma": data["lemmas"],
                    "Stem": data["stems"]
                })
                st.dataframe(table_data, use_container_width=True, height=300)

            col1, col2 = st.columns(2)

            with col1:
                # POS Tags
                with st.expander("🧠 POS Tags"):
                    pos_df = pd.DataFrame(data["pos_tags"], columns=["Token", "POS Tag"])
                    st.dataframe(pos_df, use_container_width=True, height=300)

            with col2:
                # NER Tags
                with st.expander("🏷️ Named Entities"):
                    if data["named_entities"]:
                        ner_df = pd.DataFrame(data["named_entities"], columns=["Entity", "Label"])
                        st.dataframe(ner_df, use_container_width=True, height=300)
                    else:
                        st.info("No named entities found.")

        else:
            st.error("❌ API request failed. Is the FastAPI server running?")

# --- TAB 2: Stem vs Lemma Comparison ---
with tab2:
    st.subheader("Compare Stemming vs Lemmatization")
    default_input = "running, flies, studies, better, cats, playing, talked, studies, swimming, worse"
    example_words = st.text_input("Enter comma-separated words:", default_input)

    if st.button("Compare Stem & Lemma"):
        words = [w.strip() for w in example_words.split(",")]
        res = requests.post(f"{API_URL}/compare_stem_lemma", json={"words": words})

        if res.status_code == 200:
            comp_df = pd.DataFrame(res.json())
            st.dataframe(comp_df, use_container_width=True, height=400)
        else:
            st.error("❌ API request failed.")

# --- TAB 3: TF-IDF and Embeddings Visualization ---
with tab3:
    st.subheader("TF-IDF Embeddings Visualization")
    
    # Text area for document input
    st.write("Enter multiple documents (one per line):")
    documents_text = st.text_area(
        "Example documents:", 
        """Machine learning is a subset of artificial intelligence.
Natural language processing deals with text and speech.
Deep learning uses neural networks with many layers.
Python is a popular programming language for data science.""",
        height=150
    )
    
    # Split documents
    documents = [doc.strip() for doc in documents_text.splitlines() if doc.strip()]
    
    # Get document count for setting max dimensions
    doc_count = len(documents)
    max_dims = max(min(doc_count - 1, 10), 2) if doc_count > 1 else 2
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider("Number of PCA Dimensions", 
                               min_value=2, max_value=max_dims, 
                               value=min(2, max_dims))
    
    with col2:
        viz_dimensions = st.selectbox(
            "Visualization Dimensions",
            [2, 3] if n_components >= 2 else [2],
            format_func=lambda x: f"{x}D Visualization"
        )
    
    if st.button("Generate TF-IDF Embeddings") and documents:
        with st.spinner("Generating embeddings..."):
            res = requests.post(
                f"{API_URL}/tfidf", 
                json={
                    "documents": documents,
                    "reduction_method": "pca",  # Always use PCA
                    "n_components": n_components
                }
            )
        
        if res.status_code == 200:
            data = res.json()
            
            # Display visualization
            st.success("Embeddings generated successfully!")
            
            # Prepare data for visualization
            reduced_data = np.array(data["reduced_embeddings"])
            
            # Create DataFrame for plotting
            plot_data = pd.DataFrame()
            for i in range(min(viz_dimensions, reduced_data.shape[1])):
                plot_data[f'PC{i+1}'] = reduced_data[:, i]
            
            # Add document labels
            plot_data["Document"] = [f"Doc {i+1}" for i in range(len(documents))]
            
            # Create visualization based on visualization dimensions
            if viz_dimensions == 2:
                # Create 2D scatter plot using first two dimensions
                fig = px.scatter(
                    plot_data, 
                    x='PC1', 
                    y='PC2',
                    text="Document",
                    title=f"Document Embeddings (PCA - {n_components} components)",
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
                )
            else:  # 3D visualization
                # Create 3D scatter plot using first three dimensions
                fig = px.scatter_3d(
                    plot_data,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    text="Document",
                    title=f"Document Embeddings (PCA - {n_components} components)",
                    labels={
                        'PC1': 'Principal Component 1',
                        'PC2': 'Principal Component 2',
                        'PC3': 'Principal Component 3'
                    }
                )
            
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display variance explained if PCA was used
            st.info(f"Note: Only the first {viz_dimensions} principal components are visualized, but {n_components} were calculated.")
            
            # Display top terms for each document
            st.subheader("Top TF-IDF Terms by Document")
            for i, (doc, terms) in enumerate(zip(documents, data["top_terms"])):
                with st.expander(f"Document {i+1}"):
                    st.write(f"**Text:** {doc}")
                    terms_df = pd.DataFrame(terms, columns=["Term", "TF-IDF Score"])
                    st.dataframe(terms_df, use_container_width=True)
        else:
            st.error(f"❌ API request failed with status code {res.status_code}.")
            st.error(res.text)

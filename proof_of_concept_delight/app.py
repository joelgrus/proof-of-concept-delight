import streamlit as st

from proof_of_concept_delight.model import Model

OUTPUT_DIR = 'saved_model'

model = Model.load(OUTPUT_DIR)

text = st.text_input("text")

if text:
    cats = model.predict(text)
    data = sorted(cats.items(), key=lambda pair: pair[1], reverse=True)
    st.dataframe(data)
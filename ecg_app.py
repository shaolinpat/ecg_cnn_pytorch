import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("ECGCoveNet: ECG Classification Demo")

uploaded_file = st.file_uploader("Upload ECG CSV", type=["CSV"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your data:", df.head())

    fig, ax = plt.subplots()
    ax.plot(df.iloc[:, 0], df.iloc[:, 1])
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title("ECG Signal")
    st.pyplot(fig)
    st.write("File received", uploaded_file.name)

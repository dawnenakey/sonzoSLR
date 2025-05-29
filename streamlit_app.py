"""
Streamlit app for data annotation and analysis (OAK camera removed).
"""
import streamlit as st

def main():
    st.title("Data Annotation and Analysis Platform")
    st.write("This app is now focused on annotation and analysis. OAK camera features have been removed.")
    st.info("Upload your data or start annotating using the tools below.")
    uploaded_file = st.file_uploader("Upload a data file for annotation or analysis")
    if uploaded_file:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        # Add your annotation/analysis logic here

if __name__ == "__main__":
    main() 
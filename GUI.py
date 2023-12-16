
import streamlit as st


class Gui:
    def _init_(self):
        self.judul = " "
        self.abstrak = " "
        # self.golden_keyword = []
        self.confusion_matrix = []
        self.jumlah_key = 5

    def input(self):
        self.judul = st.text_area("📌Masukkan Judul:")
        self.abstrak = st.text_area("📌Masukkan Abstrak:")
        # self.golden_keyword = st.text_area("Masukkan Golden keyphrase :")
        # self.golden_keyword = self.golden_keyword.split(";")
        self.jumlah_key = st.slider("💡Masukkan Jumlah Key")

       

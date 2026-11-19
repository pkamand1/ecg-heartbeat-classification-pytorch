"""Download the MIT-BIH Arrhythmia Database (about 100 MB) from PhysioNet.

Usage: python src/download_data.py
"""
import wfdb

wfdb.dl_database("mitdb", "data/mitdb")

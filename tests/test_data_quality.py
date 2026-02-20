import pytest
import pandas as pd
import numpy as np
from src.utils.data_quality import print_data_quality

def test_data_quality_with_missing():
    """Test data quality check detects missing values"""
    df = pd.DataFrame({
        'age': [20, np.nan, 22],
        'sleep': [7, 8, np.nan]
    })
    result = print_data_quality(df, return_dict=True)
    assert len(result['missing_pct']) > 0

def test_data_quality_with_inf():
    """Test data quality check detects infinite values"""
    df = pd.DataFrame({
        'age': [20, np.inf, 22],
        'sleep': [7, 8, 9]
    })
    result = print_data_quality(df, return_dict=True)
    assert result['has_inf'] == True

def test_data_quality_clean_data():
    """Test data quality with clean data"""
    df = pd.DataFrame({
        'age': [20, 21, 22],
        'sleep': [7, 8, 9]
    })
    result = print_data_quality(df, return_dict=True)
    assert len(result['missing_pct']) == 0
    assert result['has_inf'] == False

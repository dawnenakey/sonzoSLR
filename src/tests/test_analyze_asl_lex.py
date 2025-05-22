import os
from src.data.analyze_asl_lex import analyze_asl_lex_data

def test_analyze_asl_lex_data_runs():
    analyze_asl_lex_data('src/data/asl_lex')
    assert os.path.exists('reports')
    # Check that at least one plot was created
    files = os.listdir('reports')
    assert any(f.endswith('.png') for f in files) 
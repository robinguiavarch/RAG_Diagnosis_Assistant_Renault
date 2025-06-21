"""
Module d'évaluation - Exports simplifiés
"""

from .response_evaluator import ResponseEvaluator

def create_response_evaluator():
    """Fonction utilitaire pour usage rapide"""
    return ResponseEvaluator()
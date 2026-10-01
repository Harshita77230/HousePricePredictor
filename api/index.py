"""
api/index.py — Vercel WSGI entry point for House Price Predictor
Vercel imports this module and calls the `app` object as a WSGI handler.
"""
import sys
import os

# Make the project root importable so Flask can find templates/ and models/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

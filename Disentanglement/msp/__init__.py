"""Standalone MSP-Podcast disentanglement pipeline.

Self-contained: reuses the shared model/losses as read-only libraries but owns its
config, data, trainer and gradient-conflict handling.  Does not modify or import the
legacy train.py / run.py / config.py.
"""

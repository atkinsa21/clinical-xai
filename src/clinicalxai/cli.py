"""ClinicalXAI Command Line Interface (CLI)"""

import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(prog="clinicalxai")
    parser.parse_args(argv)
    return 0

#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append("./src/ControllerLearning/models/")
import time

#import rospy
from bodyrate_learner import BodyrateLearner

from config.settings import create_settings


def main():
    parser = argparse.ArgumentParser(description='Train Network')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='train')

    learner = BodyrateLearner(settings=settings)
    learner.train()


if __name__ == "__main__":
    main()

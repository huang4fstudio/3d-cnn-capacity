#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json


# [START datalabeling_create_annotation_spec_set_beta]
def create_annotation_spec_set(project_id):
    """Creates a data labeling annotation spec set for the given
    Google Cloud project.
    """
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()

    project_path = client.project_path(project_id)

    input_data = json.loads('["Apply Eye Makeup", "Apply Lipstick", "Archery", "Baby Crawling", "Balance Beam", "Band Marching", "Baseball Pitch", "Basketball Shooting", "Basketball Dunk", "Bench Press", "Biking", "Billiards Shot", "Blow Dry Hair", "Blowing Candles", "Body Weight Squats", "Bowling", "Boxing Punching Bag", "Boxing Speed Bag", "Breaststroke", "Brushing Teeth", "Clean and Jerk", "Cliff Diving", "Cricket Bowling", "Cricket Shot", "Cutting In Kitchen", "Diving", "Drumming", "Fencing", "Field Hockey Penalty", "Floor Gymnastics", "Frisbee Catch", "Front Crawl", "Golf Swing", "Haircut", "Hammer Throw", "Hammering", "Handstand Pushups", "Handstand Walking", "Head Massage", "High Jump", "Horse Race", "Horse Riding", "Hula Hoop", "Ice Dancing", "Javelin Throw", "Juggling Balls", "Jump Rope", "Jumping Jack", "Kayaking", "Knitting", "Long Jump", "Lunges", "Military Parade", "Mixing Batter", "Mopping Floor", "Nun chucks", "Parallel Bars", "Pizza Tossing", "Playing Guitar", "Playing Piano", "Playing Tabla", "Playing Violin", "Playing Cello", "Playing Daf", "Playing Dhol", "Playing Flute", "Playing Sitar", "Pole Vault", "Pommel Horse", "Pull Ups", "Punch", "Push Ups", "Rafting", "Rock Climbing Indoor", "Rope Climbing", "Rowing", "Salsa Spins", "Shaving Beard", "Shotput", "Skate Boarding", "Skiing", "Skijet", "Sky Diving", "Soccer Juggling", "Soccer Penalty", "Still Rings", "Sumo Wrestling", "Surfing", "Swing", "Table Tennis Shot", "Tai Chi", "Tennis Swing", "Throw Discus", "Trampoline Jumping", "Typing", "Uneven Bars", "Volleyball Spiking", "Walking with a dog", "Wall Pushups", "Writing On Board", "Yo Yo"]')

    annotation_specs = [
        datalabeling.types.AnnotationSpec(
            display_name=n,
            description=n,
        ) for n in input_data
    ]

    annotation_spec_set = datalabeling.types.AnnotationSpecSet(
        display_name='UCF-101 Full Label Set',
        description='Labels for the UCF-101 dataset',
        annotation_specs=annotation_specs
    )

    response = client.create_annotation_spec_set(
        project_path, annotation_spec_set)

    # The format of the resource name:
    # project_id/{project_id}/annotationSpecSets/{annotationSpecSets_id}
    print('The annotation_spec_set resource name: {}'.format(response.name))
    print('Display name: {}'.format(response.display_name))
    print('Description: {}'.format(response.description))
    print('Annotation specs:')
    for annotation_spec in response.annotation_specs:
        print('\tDisplay name: {}'.format(annotation_spec.display_name))
        print('\tDescription: {}\n'.format(annotation_spec.description))

    return response
# [END datalabeling_create_annotation_spec_set_beta]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--project-id',
        help='Project ID. Required.',
        required=True
    )

    args = parser.parse_args()

    create_annotation_spec_set(args.project_id)
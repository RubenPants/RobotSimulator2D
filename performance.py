import argparse
from timeit import timeit


def test_intersection():
    # Do measures
    time_cy = timeit('test_intersection_cy.main()',
                     setup='from environment.cy_entities import test_intersection_cy',
                     number=1000)
    time_py = timeit('test_intersection.main()',
                     setup='from tests import test_intersection',
                     number=1000)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the intersection test".format(round(time_py / time_cy, 2)))


def test_drive():
    # Do measures
    time_cy = timeit('test_drive_cy.main()',
                     setup='from environment.cy_entities import test_drive_cy',
                     number=10)
    time_py = timeit('test_drive.main()',
                     setup='from tests import test_drive',
                     number=10)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the game test".format(round(time_py / time_cy, 2)))


def test_sensors():
    # Do measures
    time_cy = timeit('test_sensors_cy.main()',
                     setup='from environment.cy_entities import test_sensors_cy',
                     number=1000)
    time_py = timeit('test_sensors.main()',
                     setup='from tests import test_sensors',
                     number=1000)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the game test".format(round(time_py / time_cy, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval', type=str, default='all')
    args = parser.parse_args()
    
    x = args.eval
    if x == 'all' or x == 'intersection':
        test_intersection()
    if x == 'all' or x == 'drive':
        test_drive()
    if x == 'all' or x == 'sensors':
        test_sensors()

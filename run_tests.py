"""
run_tests.py

Run all the tests.
"""
from tqdm import tqdm

from tests.drive_test import main as drive_tests
from tests.gru_test import main as gru_tests
from tests.intersection_test import main as intersection_tests
from tests.sensors_test import main as sensor_tests
from tests.simple_network_test import main as simple_network_tests

if __name__ == '__main__':
    pbar = tqdm(total=5, desc="Testing")
    success, fail = 0, 0
    
    r = drive_tests()
    success += r[0]
    fail += r[1]
    if r[1] > 0: print("> Test failed at drive_test!")
    pbar.update()
    
    r = gru_tests()
    success += r[0]
    fail += r[1]
    if r[1] > 0: print("> Test failed at gru_test!")
    pbar.update()
    
    r = intersection_tests()
    success += r[0]
    fail += r[1]
    if r[1] > 0: print("> Test failed at intersection_test!")
    pbar.update()
    
    r = sensor_tests()
    success += r[0]
    fail += r[1]
    if r[1] > 0: print("> Test failed at sensor_test!")
    pbar.update()
    
    r = simple_network_tests()
    success += r[0]
    fail += r[1]
    if r[1] > 0: print("> Test failed at simple_network_test!")
    pbar.update()
    
    print("Testing overview:")
    print(f"\tsucceeded: {success}")
    print(f"\tfailed: {fail}")

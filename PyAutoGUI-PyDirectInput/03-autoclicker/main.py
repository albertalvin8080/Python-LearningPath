import pydirectinput
import random
from time import sleep

print("Commencing...")

running = True
while running:
    # sleep(random.uniform(3, 3))
    sleep(random.uniform(60, 120))
    pydirectinput.click()
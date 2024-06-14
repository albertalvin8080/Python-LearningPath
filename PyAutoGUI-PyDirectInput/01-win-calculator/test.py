import pyautogui
import pydirectinput
from time import sleep

# while True:
#     p = pyautogui.position()
#     print(p)
#     sleep(1)

images_path = "./images"
# taskbar_region = (0, 1000, 1920, 80)

# print(pyautogui.locateCenterOnScreen(image=f"{images_path}/calculator.png"))

# Locate the calculator icon on the taskbar
point = pyautogui.locateCenterOnScreen(
    image=f"{images_path}/calculator.png", 
    # region=taskbar_region, 
    confidence=0.9  # Adjust confidence level if needed
)

print(point)
print(point.x, point.y)
pyautogui.moveTo(point.x, point.y, duration=2)
# pydirectinput.moveTo(point.x, point.y, duration=5) # Doesnt support 'duration' in the same way as pyautogui
# sleep(1)

pydirectinput.click()
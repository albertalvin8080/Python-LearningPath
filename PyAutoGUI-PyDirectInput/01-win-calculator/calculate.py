import pyautogui
import pydirectinput
from time import sleep

numbers_arr = ["", "/1.png", "/2.png"]
symbols_arr = ["/calculator.png", "/sum.png", "/equals.png"]
numbers_path = "./images/numbers"
symbols_path = "./images/symbols"

x, y = pyautogui.locateCenterOnScreen(image=symbols_path + symbols_arr[0], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
sleep(1)

x, y = pyautogui.locateCenterOnScreen(image=numbers_path + numbers_arr[1], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
# sleep(1)

x, y = pyautogui.locateCenterOnScreen(image=numbers_path + numbers_arr[2], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
# sleep(1)

x, y = pyautogui.locateCenterOnScreen(image=symbols_path + symbols_arr[1], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
# sleep(1)

x, y = pyautogui.locateCenterOnScreen(image=numbers_path + numbers_arr[2], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
# sleep(1)

x, y = pyautogui.locateCenterOnScreen(image=symbols_path + symbols_arr[2], confidence=0.9)
pyautogui.click(x, y, duration=0.5)
# sleep(1)

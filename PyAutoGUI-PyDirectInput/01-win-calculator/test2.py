import pyautogui
import pydirectinput
from time import sleep

pyautogui.hotkey("win", "r")
sleep(1)
pydirectinput.write("%temp%")
pyautogui.hotkey("enter")
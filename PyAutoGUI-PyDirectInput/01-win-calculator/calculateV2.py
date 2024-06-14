import re
import pymsgbox
import pyautogui
import pydirectinput
from time import sleep

numbers_path = "./images/numbers"
symbols_path = "./images/symbols"
raw_symbols = {
    "/": "/div.png",
    "+": "/sum.png",
    "-": "/sub.png",
    "*": "/mult.png",
    "=": "/equals.png",
    "calc": "/calculatorV2.png",
    "maximize": "/maximize.png",
}

numbers_arr = [f"{numbers_path}/{n}.png" for n in range(10)]
symbols_dict = {key: f"{symbols_path}{value}" for key, value in raw_symbols.items()}


def validate_operation(raw_operation):
    # Removing whitespaces
    parsed_operation = re.sub(r"\s+", "", raw_operation)

    # Validating the operation format
    if not re.search(r"^[0-9]+[+\-*/][0-9]+$", parsed_operation):
        raise ValueError("Invalid character or operation format provided.")

    return list(parsed_operation)


def perform_calculation(list_operation, confidence):
    print("* Running...")
    sleep(0.5)
    pyautogui.hotkey("win")
    sleep(1)

    # pyautogui.write("calculator") # Writtes too fast, so it needs a sleep()
    # sleep(1)
    pydirectinput.write("calculator")

    x, y = pyautogui.locateCenterOnScreen(symbols_dict["calc"], confidence=confidence)
    pyautogui.moveTo(x, y, duration=0.5)
    pydirectinput.click()
    sleep(1.5)

    # x, y = pyautogui.locateCenterOnScreen(symbols_dict["maximize"])
    # pyautogui.moveTo(x, y, duration=1)
    # pydirectinput.click()
    # sleep(1)

    for char in list_operation:
        button = None  # Not really necessary.

        if char.isdigit():
            button = numbers_arr[int(char)]
        else:
            button = symbols_dict[char]

        # print(button)
        x, y = pyautogui.locateCenterOnScreen(button, confidence=confidence)
        pyautogui.moveTo(x, y, duration=0.2)
        pydirectinput.click()

    print("* Completed.")

if __name__ == "__main__":

    confidence = 0.95

    # raw_operation = input("Give me an operation (Ex: 12*15):\n> ")
    raw_operation = pymsgbox.prompt("Give me an operation (Ex: 12*15)", "CalculateV2")
    list_operation = validate_operation(raw_operation)
    list_operation.append("=")
    # print(list_operation)

    perform_calculation(list_operation, confidence)

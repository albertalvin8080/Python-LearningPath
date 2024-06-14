# numbers = list(map(str, range(10)))
# print(numbers)

numbers_path = "./images/numbers"
symbols_path = "./images/symbols"

raw_symbols = {
    "/": "/div.png",
    "+": "/sum.png",
    "-": "/sub.png",
    "*": "/mult.png",
    "=": "/equals.png",
    "calc": "/calculatorV2.png",
}

numbers_arr = [f"{numbers_path}/{n}.png" for n in range(10)]
symbols_dict = {key: f"{symbols_path}{value}" for key, value in raw_symbols.items()}

print(numbers_arr, "\n\n", symbols_dict)
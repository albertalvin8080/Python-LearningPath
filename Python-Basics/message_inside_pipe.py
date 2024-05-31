def create_message_pipe(message: str, pipe_size: int = 40, decorator: str = "*"):
    message_len = len(message)

    if message_len >= pipe_size or pipe_size - 2 == message_len:
        pipe_size = message_len + 4;

    # Checks if the number of whitespaces will be even
    if (pipe_size - message_len) % 2 != 0:
        pipe_size += 1

    subtraction = pipe_size - message_len

    print()
    print(decorator * pipe_size) # string replication
    for i in range(subtraction+1):
        if i == 0 or i == subtraction:
            print(decorator, end="")
        elif i != subtraction / 2:
            print(" ",end="")
        else:
            print(message, end="")
    print()
    print(decorator * pipe_size) 

if __name__ == "__main__":
    message = "Today is a great day to"
    create_message_pipe(message)
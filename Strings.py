string = "  The Primeval Current is a flow of energy that connects to the void.  "

print(string.strip())
print(len(string.strip()))
print(string.replace(" ","-"))

text = string.split(None)
print(text[2:4])
# print(text[1])
# print(string)

evento = "{} de {} de {} às {} horas"
print(evento.format(3,"Maio",2019,"17"))

str1 = 'Super Crucible'
#str1[1]='x'
#str1 = str1[0:4] + 'x' + str1[4+1:len(str1)]
str1 = str1[:4] + 'x' + str1[4+1:]
print(str1)
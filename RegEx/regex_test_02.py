import re

string = '''
hoje eu comi muitas frutas:
mamão, melão, maçã e laranja...

Today's a great day. We don't have to worry about escaping from Godrick anymore.
Except for you, obviously.

testa.do@gmail.com
trombadinha@domain.gov
'''

pattern = re.compile('\w*e[^\w]')
matches = pattern.finditer(string)

# subbed_string = re.split(',', string)
# print(subbed_string)

for match in matches:
    print(match)
import re

texto = '''
Estamos aqui reunidos para decifrar
o código de Midas.

RxJ00-9000
Jptc8-9900

Precisamos de voluntários e ideias,
estamos ficando sem tempo.

Enviem qualquer pista para o e-mail:
entusi.asta@gmail.gov
testanto..a@hotmail.dev
'''

pattern = re.compile('([\w.]+)@(\w+)(\.\w+)')

matches = pattern.finditer(texto)

for match in matches:
    print(match.group(0))

# subbed_texto = pattern.sub(r'*substituindo*',texto)
subbed_texto = pattern.sub(r'\3\2@\1',texto)
print(subbed_texto)
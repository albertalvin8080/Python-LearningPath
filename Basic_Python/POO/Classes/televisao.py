class Televisao:
    __canal_limite = (1, 100)
    __volume_limite = (1, 10)

    def __init__(self) -> None:
        self._canal = 1
        self._volume = 5

    @classmethod
    def construir_TV(cls): # factory method
        return cls()
    
    @staticmethod
    def mostrar_limites():
        return (Televisao.__canal_limite, Televisao.__volume_limite)
    
    def mudar_canal(self, numero: int):
        if numero < self.__canal_limite[0] or numero > self.__canal_limite[1]:
            raise CanalInvalido(f'O canal deve estar entre {self.__canal_limite[0]} e {self.__canal_limite[1]}')
        else:
            self._canal = numero
            print(f'Canal mudou para {numero}')
    
    def mudar_volume(self, unidades: int):
        if self._volume + unidades < self.__volume_limite[0]:
            self._volume = self.__volume_limite[0]

        elif self._volume + unidades > self.__volume_limite[1]:
            self._volume = self.__volume_limite[1]

        else:
            self._volume += unidades

        print(f'Volume mudou para {self._volume}')
    
class CanalInvalido(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)

if __name__ == '__main__':
    tv = Televisao.construir_TV()
    print(tv.mostrar_limites())
    tv.mudar_canal(4)
    tv.mudar_canal(-100)
    # tv.mudar_canal(101)
    tv.mudar_volume(2)
    tv.mudar_volume(-10)
    tv.mudar_volume(-10)
    tv.mudar_volume(+15)
    tv.mudar_volume(-3)
    tv.mudar_volume(-2)

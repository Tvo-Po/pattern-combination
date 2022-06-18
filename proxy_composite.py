"""Демонстрирует взаимодействие таких структурных шаблонов проектирования, как компоновщик и прокси.
При запуске скрипта создается 8 компонентов (телефон/модель/общая группа), каждый из которых
обернут объектом прокси-класса. Далее дерево, полученное через шаблон "Компановщик", линеаризуется
с помощью функции linearize_composite. После чего у каждого объекта вызываются общие методы
"Компановщика", результат которых выводится в консоль, т.к. они обернуты прокси. 


Классы:

    PhoneComponent
    Phone
    PhoneModel
    PhoneComponentProxy

Функции:

    linearize_composite

Ошибки:

    UnknownRoleError
"""


from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from typing import List


class PhoneComponent(ABC):
    """
    Базовый абстрактный класс для реализации компонентов
    шаблона Компоновщик на примере смартфонов
    Аргументы:
        _tree_level: уровень объекта в дереве компонентов (0 - корень)
    """
    _tree_level = 0

    def __init__(self, name: str):
        """
        Все компоненты обладают именем, поэтому
        оно инициализируется в базовом классе
        """
        self._name = name

    @property
    def name(self) -> str:
        """Возвращает название компонента"""
        return self._name

    @abstractmethod
    def get_price(self) -> int:
        """Абстракный метод для получения цены компонента"""
        pass

    def is_composite(self) -> bool:
        """
        Метод, сообщающий, является ли компонент составным
        по умолчанию компонент не является составным
        """
        return False

    def set_tree_level(self, tree_level: int):
        """Сеттер для аргумента _tree_level"""
        self._tree_level = tree_level


class Phone(PhoneComponent):
    """
    Конечный объект в шаблоне Компоновщик,
    олицетворяет определенную модель смартфона
    """

    def __init__(self, name: str, price: int):
        """Помимо имени конечный компонент инициализируется ценой"""
        self._price = price
        super().__init__(name)

    def get_price(self) -> int:
        """Возвращает собственную цену"""
        return self._price


class PhoneModel(PhoneComponent):
    """
    Составной объект Компоновщика, может
    представлять марку, группу телефонов
    """

    def __init__(self, name: str, phones: List[PhoneComponent]):
        """Помимо имени инициализируется списком компонентов"""
        self._phones = phones
        super().__init__(name)

    def get_price(self) -> int:
        """
        Получает цену, вызывая у принадлежащих составному компоненту
        объектов метод get_price и суммируя полученные значения
        """
        return reduce(
            lambda total_price, phone: total_price + phone.get_price(),
            self._phones, 0
        )

    def is_composite(self) -> bool:
        """Сообщает, что компонент является составным"""
        return True

    @property
    def components(self) -> List[PhoneComponent]:
        """Возвращает список своих дочерних компонентов"""
        return self._phones


class RoleEnum(Enum):
    """Перечисление для ролей класса-прокси"""
    MANAGER = 'manager'
    CLIENT = 'client'


class UnknownRoleError(Exception):
    """Ошибка, вызываемая при передачи неизвестной роли"""
    pass


class PhoneComponentProxy:
    """
    Класс-прокси для работы с объектами классов шаблона Компоновщик
    Выводит в консоль данные, полученные с методов этих объектов, в
    зависимости от уровня доступа
    Аргументы:
        _level_char: символ для отображения уровня компонента
    """
    _level_char = '\t'

    def __init__(self, component: PhoneComponent, role: RoleEnum):
        """Получает на вход роль пользователя и оборачиваемый компонент"""
        try:
            role in RoleEnum
        except TypeError:
            raise UnknownRoleError(f'Role {role} is invalid.')
        self._role = role
        self._component = component

    @property
    def tree_level(self) -> int:
        """Возвращает уровень оборачиваемого компонента в дереве"""
        return self._component._tree_level

    def set_tree_level(self, tree_level: int):
        """Сеттер для уровня оборачиваемого компонента в дереве"""
        return self._component.set_tree_level(tree_level)

    @property
    def name(self) -> str:
        """Возвращает название компонента и печатает его в консоль"""
        print(f'{self._level_char * self.tree_level}Name: {self._component.name}')
        return self._component.name

    @property
    def price(self) -> str:
        if self._role is RoleEnum.MANAGER:
            return f'{self._price}₽'
        elif self._role is RoleEnum.CLIENT:
            return '<view not allowed>'

    def get_price(self, is_outer_call: bool = False) -> int:
        """
        Проверяет, имеется ли атрибут с ценой товара в классе, если нет,
        то проверяет роль пользователя прокси, если это менеджер, то
        получает цену компонента и присваивает ее в этот атрибут, в
        случае, когда вызов является внешним - печатает в консоль цену
        компонента (сделано для исключение вывода в консоль вложенных вызывов
        get_price). Для клиента просмотр цены запрещен.
        """
        if not hasattr(self, '_price'):
            self._price = self._component.get_price()
        if is_outer_call:
            print(f'{self._level_char * self.tree_level}Price: {self.price}')
        return self._price

    def is_composite(self, is_inner_call: bool = False) -> bool:
        """
        Сообщает, является ли оборачиваемый компонент составным, если
        вызов не внутренний, то печатает заголовок для компонента в
        зависимости от возвращаемой характеристики
        """
        if not is_inner_call:
            print(f'{self._level_char * self.tree_level}===== {"Model" if self._component.is_composite() else "Phone"}  =====')
        return self._component.is_composite()

    @property
    def components(self) -> List[PhoneComponent]:
        """
        Возвращает список дочерних компонентов оборачиваемого объекта,
        если он составной, иначе возвращает пустой список
        """
        if self._component.is_composite():
            return self._component.components
        else:
            return []


def linearize_composite(
        components: List[PhoneComponent],
        linearization: List[PhoneComponent] = [],
        n: int = 1,
    ):
    """
    Функция линеаризирует все дочерних компоненты на любом уровне для
    объекта шаблона Компоновщик. Возвращает одномерный список компонентов.
    Аргументы:
        components: список дочерних компонентов объекта
        linearization: линеаризированный список, в начальном вызове должен быть
        пустым, используется в рекурсивных вызовах
        n: уровень вложенности элемента, обрабатывается через рекурсивные вызовы
    """
    for c in components:
        c.set_tree_level(n)
        if c.is_composite(is_inner_call=True):
            linearization.append(c)
            linearize_composite(c.components, linearization, n=n + 1)
        else:
            linearization.append(c)
    return linearization


if __name__ == '__main__':
    phone_1 = PhoneComponentProxy(Phone(
        'Xiaomi mi 11 lite 5G NE', 45000
    ), RoleEnum.MANAGER)
    phone_2 = PhoneComponentProxy(Phone(
        'Xiaomi MIU 5', 38000
    ), RoleEnum.MANAGER)
    model_1 = PhoneComponentProxy(PhoneModel(
        'Xiaomi', [phone_1, phone_2]
    ), RoleEnum.MANAGER)
    phone_3 = PhoneComponentProxy(Phone(
        'IPhone SE', 58000
    ), RoleEnum.CLIENT)
    phone_4 = PhoneComponentProxy(Phone(
        'IPhone 11', 80000
    ), RoleEnum.CLIENT)
    phone_5 = PhoneComponentProxy(Phone(
        'IPhone 12 Pro Max', 155000
    ), RoleEnum.CLIENT)
    model_2 = PhoneComponentProxy(PhoneModel(
        'Iphone', [phone_3, phone_4, phone_5]
    ), RoleEnum.MANAGER)
    model_3 = PhoneComponentProxy(PhoneModel(
        'Phones', [model_1, model_2]
    ), RoleEnum.MANAGER)
    composite_list = [model_3, *list(linearize_composite(model_3.components))]
    for component in composite_list:
        component.is_composite()
        component.name
        component.get_price(is_outer_call=True)

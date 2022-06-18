"""Демонстрирует взаимодействие таких поведенческих шаблонов проектирования, как цепь обязанностей
и интерпретатор. При запуске скрипта имитируется пользовательский ввод 6 телефонов, которые далее
интерпретируются в объекты датакласса Phone и выводятся в консоль. Некоторые строки пользовательского
ввода заведомо написаны некорректно с точки зрения интерпретатора, тем самым они вывывают ошибки
интерпретации, которые перехватываются и выводятся в консоль. Далее сформированные объекты телефонов
валидируются, при этом найденные ошибки выводятся в консоль.

Метаклассы:

    AbstractStringMappingInterpretatorMetaclass
    AbstractStringNumberWithMeasureInterpretatorMetaclass

Классы:

    Phone
    AbstractInterpretator
    StringInterpretator
    IntegerInterpretator
    StringMappingInterpretator
    StringNumberWithMeasureInterpretator
    ModelInterpretator
    SeriesInterpretator
    ResolutionInterpretator
    MemoryInterpretator
    RamInterpretator
    CameraInterpretator
    PhoneInterpretator
    PhoneAbstractValidator
    PhoneModelValidator
    PhoneSeriesValidator
    PhoneResolutionValidator
    PhoneMemoryValidator

Функции:

    get_validators_chain

Ошибки:

    InterpretatorAttributeError
    InterpretationError
    ValidationError
"""


from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from ctypes import Union
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    DefaultDict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    TYPE_CHECKING,
)


if TYPE_CHECKING:
    from toi7 import PhoneAbstractValidator


class InterpretatorAttributeError(Exception):
    """Ошибка, вызываемая при неверном объявлении атрибутов в классах интерпритатора"""
    pass


class InterpretationError(Exception):
    """Ошибка, вызываемая при неверном формате данных, передаваемых интерпритатору"""
    
    def __init__(
        self, *args,
        message: Optional[str | list] = None,
        errors: Optional[DefaultDict[str, list]] = None,
    ) -> None:
        """Если на вход передан словарь ошибок, формирует из него сообщение об ошибках"""
        if errors:
            message = self.format_errors_dict(errors)
        self.message = message
        super().__init__(self.message)
    
    def add_to_errors_dict(self, errors_dict: DefaultDict[str, list], attr: str) -> None:
        """Добавляет в словарь ошибок к переданному полю сообщение, в зависимости от ее типа"""
        if isinstance(self.message, str):
            errors_dict[attr].append(self.message)
        elif isinstance(self.message, list):
            errors_dict[attr].extend(self.message)
    
    def format_errors_dict(self, errors: DefaultDict[str, list]) -> str:
        """Форматирует словрь с ошибками под строку"""
        message = '\n'
        for attr, error in errors.items():
            if isinstance(error, list):
                message += '{}: {}\n'.format(attr, '\n'.join(error))
            elif isinstance(error, str):
                message += '{}: {}\n'.format(attr, error)
        return message


@dataclass
class Phone:
    """
    Программное представление телефона
    Также содержит хранилище всех марок и моделей
    """
    model: str
    series: str
    resolution: int
    memory: int
    ram: int
    camera: float
    
    class Model(NamedTuple):
        xiaomi = 'Xiaomi'
        iphone = 'IPhone'
        meizu = 'Meizu'
    
    class Series(NamedTuple):
        mi5 = 'MI 5'
        redmi = 'Redmi'
        miu = 'MIU'
        se = 'SE'


class AbstractInterpretator(ABC):
    """Базовый абстрактный класс для интерпретатора"""
    
    @abstractmethod
    def interpret(self) -> Any:
        pass


class StringInterpretator(AbstractInterpretator):
    """Базовый класс для строкового интерпретатора"""
    
    def __init__(self, interpretated: str) -> None:
        self._interpretated = interpretated
    
    def interpret(self) -> str:
        return self._interpretated


class IntegerInterpretator(StringInterpretator):
    """Базовый класс для целочисленного интерпретатора"""
    
    def interpret(self) -> str:
        """Переводит строку в целое число"""
        try:
            value = int(self._interpretated)
        except ValueError:
            raise InterpretationError(
                message=f'Значение <{self._interpretated}> не является целочисленным.'
            )
        return value


class AbstractStringMappingInterpretatorMetaclass(ABCMeta):
    """Метакласс для валидации атрибутов дочерних классов StringMappingInterpretator"""
    
    def __new__(cls, name, bases, attrs):
        if not name == 'StringMappingInterpretator':
            cls.validate_mapping(name, attrs)
        return super().__new__(cls, name, bases, attrs)
    
    @classmethod
    def validate_mapping(cls, name: str, attrs: dict) -> None:
        """
        Дочерние классы StringMappingInterpretator должны иметь атрибут
        mapping_name и ru_mapping_name, в которых содержится англ./рус.
        название параметра, по которому осуществляется отображение.
        Этот атрибут должен быть формата <mapping_name>_mapping и,
        соответственно, должен являться отображением.
        """
        mapping_name = attrs.get('mapping_name', None)
        if mapping_name is None:
            raise InterpretatorAttributeError(
                f'Атрибут <mapping_name> является обязательным для класса {name}.'
            )
        if not 'ru_mapping_name' in attrs:
            raise InterpretatorAttributeError(
                f'Атрибут <ru_mapping_name> является обязательным для класса {name}.'
            )
        mapping = attrs.get(f'{mapping_name}_mapping', None)
        if mapping is None:
            raise InterpretatorAttributeError(
                f'У класса {name} отсутствует атрибут <{mapping_name}_mapping>.'
            )
        if not isinstance(mapping, Mapping):
            raise InterpretatorAttributeError(
                f'Атрибут <{mapping_name}_mapping> класса {name} должен быть отображением.'
            )


class StringMappingInterpretator(
    StringInterpretator,
    metaclass=AbstractStringMappingInterpretatorMetaclass,
):
    
    def interpret(self) -> str:
        """Возвращает совпадение интерпретируемого значения из отображения"""
        interpretated = self._interpretated.replace(' ', '').lower()
        mapping_name = getattr(self, 'mapping_name')
        mapping = getattr(self, f'{mapping_name}_mapping')
        if not mapping.get(interpretated):
            raise InterpretationError(
                message=f'Неизвестная {getattr(self, "ru_mapping_name")} <{self._interpretated.strip()}>'
            )
        return mapping[interpretated]


class AbstractStringNumberWithMeasureInterpretatorMetaclass(ABCMeta):
    """Метакласс для валидации атрибутов дочерних классов StringNumberWithMeasureInterpretator"""
    
    def __new__(cls, name, bases, attrs):
        if not name == 'StringNumberWithMeasureInterpretator':
            cls.validate_measures(name, attrs)
            cls.validate_num_type(name, attrs)
        return super().__new__(cls, name, bases, attrs)
    
    @classmethod
    def validate_measures(cls, name, attrs) -> None:
        """
        Дочерние классы StringNumberWithMeasureInterpretator должны
        иметь атрибут measures, который должен быть итерируемым
        """
        measures = attrs.get('measures', None)
        if measures is None:
            raise InterpretatorAttributeError(
                f'У класса {name} отсутствует атрибут <measures>.'
            )
        if not isinstance(measures, Iterable):
            raise InterpretatorAttributeError(
                f'Атрибут <measures> класса {name} должен быть итерируемым.'
            )
    
    @classmethod
    def validate_num_type(cls, name, attrs) -> None:
        """
        Дочерние классы StringNumberWithMeasureInterpretator должны
        иметь атрибут num_type, который должен быть классом int или float
        """
        num_type = attrs.get('num_type', None)
        if num_type is None:
            raise InterpretatorAttributeError(
                f'У класса {name} отсутствует атрибут <num_type>.'
            )
        if not num_type in (int, float):
            raise InterpretatorAttributeError(
                f'Атрибут <num_type> класса {name} должен быть либо <int>, либо <float>.'
            )


class StringNumberWithMeasureInterpretator(
    StringInterpretator,
    metaclass=AbstractStringNumberWithMeasureInterpretatorMetaclass,
):
    
    def interpret(self) -> Union[int, float]:
        """Отчищает значение от единиц измерения и возврщает целочесленное или дробное число"""
        interpretated = self._interpretated.replace(' ', '').lower()
        measures = getattr(self, 'measures')
        str_value = reduce(
            lambda raw_value, measure: raw_value.replace(measure, ''),
            measures, interpretated
        )
        num_type = getattr(self, 'num_type')
        try:
            num_value = num_type(str_value)
        except ValueError:
            raise InterpretationError(
                message='Значение содержит неизвестную единицу измерения, либо не является {}.'.format(
                    'целым числом' if num_type is int else 'дробным числом'
                )
            )
        return num_value


class ModelInterpretator(StringMappingInterpretator):
    ru_mapping_name = 'марка'
    mapping_name = 'model'
    model_mapping = {
        'xiaomi': Phone.Model.xiaomi,
        'xiomi': Phone.Model.xiaomi,
        'xeomi': Phone.Model.xiaomi,
        'iphone': Phone.Model.iphone,
        'ifone': Phone.Model.iphone,
        'iphon': Phone.Model.iphone,
        'meizu': Phone.Model.meizu,
    }


class SeriesInterpretator(StringMappingInterpretator):
    ru_mapping_name = 'модель'
    mapping_name = 'series'
    series_mapping = {
        'mi5': Phone.Series.mi5,
        'mi_5': Phone.Series.mi5,
        'redmi': Phone.Series.redmi,
        'se': Phone.Series.se,
        'miu': Phone.Series.miu,
    }


class ResolutionInterpretator(StringInterpretator):
    """Интерпретатор, обрабатывающий разрешение экрана"""
    delimeters = ('x', '*', ' ')
    
    def interpret(self) -> int:
        """Пытается привести строку к целому числу, если не выходит,
        то пробует разбить ее по разделителю (в случае, когда разрешение
        передается в виде произведения длины и ширины дисплея), возвращая
        произведение двух полученных величин. Разделители хранятся
        в атрибуте <delimeters> данного класса."""
        interpretated = self._interpretated.lower()
        try:
            resolution = int(interpretated.replace(' ', ''))
        except ValueError:
            for delimeter in self.delimeters:
                resolution_dims = [
                    dim.replace(' ', '') for dim
                    in interpretated.split(delimeter)
                    if dim != ''
                ]
                if len(resolution_dims) == 2:
                    break
            else:
                raise InterpretationError(message='Недопустимый формат разрешения экрана.')
            errors = []
            for i, dim in enumerate(resolution_dims):
                try:
                    dim_value = IntegerInterpretator(dim).interpret()
                except InterpretationError as e:
                    errors.append(e.message)
                else:
                    resolution_dims[i] = dim_value
            if errors:
                raise InterpretationError(message=errors)
            resolution = resolution_dims[0] * resolution_dims[1]
        return resolution


class MemoryInterpretator(StringNumberWithMeasureInterpretator):
    measures = ('gb', 'mb', 'kb', 'b')
    num_type = int


class RamInterpretator(StringNumberWithMeasureInterpretator):
    measures = ('gb', 'mb', 'kb', 'b')
    num_type = int


class CameraInterpretator(StringNumberWithMeasureInterpretator):
    measures = ('mp',)
    num_type = float


class PhoneInterpretator(StringInterpretator):
    """
    Основной интерпретатор, который переводит строку формата:
        Model: <model>
        Series: <series>
        Resolution: <resolution> (568000 | 720 x 1080 | 720 * 1080 | 720 1080)
        Memory: <memory> <measure> (32 GB | 65536 mb)
        RAM: <ram> <measure> (2 GB)
        Camera: <resolution> <measure> (15.2 Mp)
    в объект датакласса Phone
    """
    
    def interpret(self) -> Phone:
        """
        Разбивает исходный текст на строки, не включая пустые, затем разбивает строку через разделитель <:>,
        слева от которого находится название атрибута, а справа значение. Потом ищет интерпритатор по названию
        атрибута и передает в него значение. Таким образом через цикл собирается словарь для создание объекта
        Phone. В случае, если другие интерпретаторы вернули ошибки, вызывает общую ошибку.
        """
        interpretated_rows = [row.strip() for row in self._interpretated.splitlines() if row.strip()]
        phone_initial_dictionary = {}
        errors = defaultdict(list)
        for i, row in enumerate(interpretated_rows):
            splitted_row = row.split(':')
            if not len(splitted_row) == 2:
                errors['non_field_errors'].append(f'Строка №{i + 1} неверного формата (Атрибут: Значение).')
                continue
            attr, raw_value = splitted_row
            interpretator_class = globals().get(f'{attr.strip().capitalize()}Interpretator')
            if not interpretator_class:
                errors[attr].append(f'Неизвестный атрибут <{attr.strip()}>.')
                continue
            interpretator = interpretator_class(raw_value)
            try:
                phone_initial_dictionary[attr.strip().lower()] = interpretator.interpret()
            except InterpretationError as e:
                e.add_to_errors_dict(errors_dict=errors, attr=attr)
        if errors:
            raise InterpretationError(errors=errors)
        return Phone(**phone_initial_dictionary)


class ValidationError(Exception):
    """Ошибка, вызываемая при передаче невалидных значений в валидатор"""
    
    def __init__(
        self, *args, message: str = '',
        errors: DefaultDict[list] = defaultdict(str),    
    ) -> None:
        """Если передан словарь с ошибками, то он преобразовывается в сообщения"""
        if not message and errors:
            self.message = None
            super().__init__(*args)
        if errors:
            if message:
                errors['non_field_errors'] = self.message
            message = self.format_errors_dict(errors)
        self.message = message
        super().__init__(self.message)
    
    def format_errors_dict(self, errors: DefaultDict[str]) -> str:
        """Форматирует словарь ошибок в строку"""
        message = '\n'
        for attr, error in errors.items():
            message += '{}: {}\n'.format(attr, error)
        return message


class PhoneAbstractValidator(ABC):
    """Абстрактный валидатор, реализуемый на шаблоне цепи ответственности"""
    
    def __init__(self) -> None:
        self._errors = {}
    
    def set_next(self, validator: PhoneAbstractValidator) -> PhoneAbstractValidator:
        """
        Устанавливает следующий валидатор и возвращает его же для удобной установки цепи
        Пример: first_validator.set_next(second_validator).set_next(third_validator)
        """
        self._next_validator = validator
        return validator
    
    @abstractmethod
    def validate(self, phone: Phone) -> None:
        """
        После валидации проверяет наличие следующего валидатора,
        если он есть и есть ошибки, передает их в него и вызывает
        у него метод validate. Если нет следующего валидатора, то
        при наличии ошибок вызывает эксепшен, иначе возвращает None.
        """
        has_next_validator = hasattr(self, '_next_validator')
        if self._errors:
            if not has_next_validator:
                raise ValidationError(errors=self._errors)
            else:
                self._next_validator._errors = self._errors
        if has_next_validator:
            return self._next_validator.validate(phone)
        return None
        


class PhoneModelValidator(PhoneAbstractValidator):
    
    def validate(self, phone: Phone) -> None:
        if phone.model == Phone.Model.iphone:
            self._errors['model'] = 'IPhone не доступен на территории РФ.'
        super().validate(phone)


class PhoneSeriesValidator(PhoneAbstractValidator):
    
    def validate(self, phone: Phone) -> None:
        if phone.series == Phone.Series.miu:
            self._errors['series'] = 'MIU сборка больше не поддерживается.'
        super().validate(phone)


class PhoneResolutionValidator(PhoneAbstractValidator):
    
    def validate(self, phone: Phone) -> None:
        if phone.resolution < 400_000:
            self._errors['resolution'] = 'Площадь экрана телефона слишком маленькая.'
        super().validate(phone)


class PhoneMemoryValidator(PhoneAbstractValidator):
    
    def validate(self, phone: Phone) -> None:
        if phone.memory < 64:
            self._errors['memory'] = 'На телефоне недостаточно памяти.'
        super().validate(phone)


def get_validators_chain() -> PhoneAbstractValidator:
    model_validator = PhoneModelValidator()
    series_validator = PhoneSeriesValidator()
    resolution_validator = PhoneResolutionValidator()
    memory_validator = PhoneMemoryValidator()
    model_validator.set_next(series_validator).set_next(resolution_validator) \
                   .set_next(resolution_validator).set_next(memory_validator)
    return model_validator


if __name__ == '__main__':
    phones_input = [
        """
            Model: Xiomi
            Series: MI_5
            Resolution: 720 * 1080
            Memory: 64 GB
            RAM: 2 GB
            Camera: 15 Mp
        """,
        """
            Model: Iphone
            Series: SE
            Resolution: 740 x 1240
            Memory: 128 GB
            RAM: 8 GB
            Camera: 32 Mp
        """,
        """
            Model: Nokia
            Series: n5420
            Resolution: 320px-450px
            Memory: 160 mb
            RAM: -
            Camera: 2,4 p
            Material: plastic
        """,
        """
            Model        :       Meizu
            Series       :       miu
            Resolution   :       480   x   620
            Memory       :       32         GB
            RAM          :       2          GB
            Camera       :       8          MP
        """,
        """
            Model: Samsung
            Series: Note 7
            Resolution: 720px x 1080px
            Memory: 32 MB
            RAM: 2 MB
            Camera: 15 MP
        """,
        """
            Model: XEOMI
            Series: REDMI
            Resolution: 640 980
            Memory: 128 GB
            RAM: 6 GB
            Camera: 16 MP
        """
    ]
    phones = []
    for phone_str in phones_input:
        try:
            phone = PhoneInterpretator(phone_str).interpret()
            phones.append(phone)
        except InterpretationError as e:
            print(f'Ошибка интерпретации:{e.message}\n')
        else:
            print(f'{phone}\n')
    print('\n======== Валидация ========\n\n')
    validator = get_validators_chain()
    for phone in phones:
        try:
            validator.validate(phone)
        except ValidationError as e:
            print(f'Ошибка валидации {phone.model} {phone.series}:{e.message}\n')

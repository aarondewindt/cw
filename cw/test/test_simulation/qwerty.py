

class Person:
    def __init__(self, name, height):
        self.name = name
        self.height = height

    def speak(self):
        print("hello I'm", self.name)


person_a = Person("foo", 1.5)
person_b = Person("bar", 1.8)

print(person_a.name)

person_b.speak()

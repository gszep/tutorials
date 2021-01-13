from numpy import *

class Altright:
    can_be_triggered = False # class level attribute

    def __init__(self,weapon):
        self.weapon = weapon

    def trigger(self,snowflake):
        snowflake.triggered = True

    def __add__(self,other):

        if type(other) is Snowflake :

            if other.triggered :
                return 'fight! fight! fight!'
            else :
                return 'invade the captol'

        elif type(other) is int :
            return Altright(self.weapon), Altright(self.weapon)

        else :
            raise NotImplementedError('+ not defined for input types')

    def __mul__(self,altright):
        return 'Elect POTATO'

class Snowflake:
    can_be_triggered = True # class level attribute

    def __init__(self,favourite_vegetable):

        self.favourite_vegetable = favourite_vegetable
        self.triggered = False # instance level attribute

    def istriggered(self):
        return self.triggered

    def __add__(self,altright):
        if self.triggered :
            return 'fight! fight! fight!'
        else :
            return 'invade the captol'

Snowflake.triggered = False # class level update; all instnces of Snowflake get updated
aoc = Snowflake('carrot') # Snowflake.__init__(aoc,'carrot')

aoc.triggered_local = False # instance level update; only the intance (aoc) gets updated
milo = Altright('memes')

type(aoc)
milo + aoc
aoc + milo # same as Snowflake.__add__(aoc,milo)

aoc.triggered
milo.trigger(aoc) # changes triggered status
aoc.istriggered()

aoc + milo # and now they fight!

milo * milo

milo + 1
milo + aoc

# methods imported to more complicated methods
array([milo,milo,milo]) + aoc

MELD = "meld"
TUCK = "tuck"
SCORE = "score"
RETURN = "return"
TRANSFER = "transfer"
REVEAL = "reveal"

ACTIONS_IN_TWO_PLACES = [TRANSFER]

HAND = "hand"
BOARD = "board"
SCOREPILE = "score pile"

BLUE = "blue"
RED = "red"
GREEN = "green"
YELLOW = "yellow"
PURPLE = "purple"

ALL = "all"
ANY = "any number of"

LEFT = "left"
RIGHT = "right"
UP = "up"

def comment(action=None, may=True, num=1, age=None, \
    top=True, color=None, icon=None, custom="", from_=HAND, to=None):
    l = ["You"]
    l.append('may' if may else 'must')
    l.append(action)
    
    l.append('a' if num == 1 else str(num))
    if top and from_ == BOARD:
        l.append("top")
    if color is not None:
        l.append(color)
    if age is not None:
        l.append(str(age))
    else:
        l.append("card" if num == 1 else "cards")
    if icon is not None:
        l.append("with a " + icon)
    l.append("from your " + from_)
    if to is not None and action in ACTIONS_IN_TWO_PLACES:
        l.append("to " + to)
    return ' '.join(l)

def splayComment(colors, direction):
    l = ["You may splay your"]
    if isinstance(colors, str):
        l.append(colors)
        l.append("cards")
    else:
        l.append(', '.join(colors[:-1]))
        l.append("or")
        l.append(colors[-1])
        l.append("cards")
    l.append(direction)
    return ' '.join(l)

if __name__ == "__main__":
    pass
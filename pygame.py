
import pygame
import pygame_widgets as pyw
from rule import Card
from agents import BaseAgent

pygame.init()

class PygameCard(Card):
    pass

class Window:
    def __init__(self, screenSize, bx=10, by=10, bw=150, bh=50):
        self.screen = pygame.set_mode(screenSize)
        self.looping = False
        self.bx = bx
        self.by = by
        self.bw = bw
        self.bh = bh
        self.left = bw+bx*2
    
    def createOnclick(returnNumber):
        def onclick():
            self.looping = False
            return returnNumber
        return onclick
    
    def createOnclicks(returnNumbers):
        return list(map(self.createOnclick, returnNumbers))
    
    def draw(self, ps):
        pass
    
    def loop(self, names, returnNumbers):
        assert len(names) == len(returnNumbers)
        self.looping = True
        buttons = pyw.ButtonArray(self.screen, x, y, width, height, (1, len(names)))
        while self.looping:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.screen.fill((255, 255, 255))

            buttons.listen(events)
            self.draw()
            buttons.draw()

            pygame.display.update()

class PygameUserAgent(BaseAgent):
    def __init__(self):
        pass

    def step(self, obs):
        pass
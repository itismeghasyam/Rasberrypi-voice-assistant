from gpiozero import Button
from signal import pause

button = Button(17)  # GPIO17 = Pin 11

button.when_pressed = lambda: print("Pressed!")
button.when_released = lambda: print("Released!")

pause()

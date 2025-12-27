"""
@pyne
"""
from pynecore.lib import script, strategy, plot, time

@script.strategy(shorttitle="TestState", overlay=False)
def main():
    # Attempt to use script.load/save for persistence
    counter = script.load("counter", 0)
    counter = counter + 1
    script.save("counter", counter)
    
    plot(counter, title="Counter")

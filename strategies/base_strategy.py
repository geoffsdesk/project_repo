from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with OHLCV data and returns the DataFrame with 
        added 'signal' column (1 for buy, -1 for sell, 0 for hold) and 
        any indicator columns.
        """
        pass

    @abstractmethod
    def get_pine_script(self) -> str:
        """
        Returns the Pine Script representation of the strategy.
        """
        pass

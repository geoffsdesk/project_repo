from strategies.base_strategy import BaseStrategy

class PineExporter:
    def export(self, strategy: BaseStrategy) -> str:
        """
        Converts a Strategy instance into Pine Script code.
        """
        header = f"// Pine Script Generated for {strategy.name}\n"
        header += "//@version=5\n"
        header += f"strategy('{strategy.name}', overlay=true)\n\n"
        
        body = strategy.get_pine_script()
        
        return header + body

import pandas as pd

class DataLoader:
    def __init__(self, source):
        self.source = source
        self.data = None

    def load_data(self):
        """
        Load data from the specified source.
        """
        # Si la source est un DataFrame pandas, on l'utilise directement
        if isinstance(self.source, pd.DataFrame):
            self.data = self.source
        else:
            # Sinon, on suppose que c'est un chemin vers un fichier CSV
            if isinstance(self.source, str) and self.source.endswith('.csv'):
                self.data = pd.read_csv(self.source)
            else:
                raise ValueError("La source doit être un DataFrame pandas ou un chemin vers un fichier CSV.")

    def get_data(self):
        return self.data
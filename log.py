class Logger:
    def __init__(self, path: str):
        self.path = path

    def log(self, status: str):
        with open(self.path, 'a') as f:
            f.write(f'{status}\n')
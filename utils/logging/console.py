from utils.logging.logger import Logger


class ConsoleLogger(Logger):
    def log(self, logs, step=None):
        if step is not None:
            print(f"step {step}:")
        for k, v in logs.items():
            print(f"{k}: {v}")
        print()
